#!/usr/bin/env python3

# Directly adapted from code by Sukanta Sen, many thanks!
# Sen et al. (2022) "Simultaneous Translation for Unsegmented Input: A Sliding Window Approach"
# https://arxiv.org/pdf/2210.09754.pdf

import sys, os
from difflib import SequenceMatcher
import argparse

from integrate import init, translate

def options():
  parser = argparse.ArgumentParser()
  parser.add_argument("-lang", default="cs", help="Language code, e.g. cs")
  parser.add_argument("-window_len", type=int, default=15)
  parser.add_argument("-config")
  parser.add_argument("-type", default="extended")
  parser.add_argument("-gpuid", type=int, default=4, help="GPU id, just one gpu is required.")
  parser.add_argument("-ratio", type=float)
  parser.add_argument("-max_history", type=float,  default=5)
  parser.add_argument("-slt", action="store_true")
  parser.add_argument("-truncate", action="store_true")
  parser.add_argument("-cache", default=None)
  parser.add_argument("-log", type=str)
  parser.add_argument("-merge_last_overlap", action="store_true")
  parser.add_argument("-shift", type=int, default=1)
  args = parser.parse_args()
  return args

def translate_window(start, end, audio_filename, temp_prefix):
    os.system(f"ffmpeg -hide_banner -loglevel error -y -i {audio_filename} -acodec copy -ss {start} -to {end} {temp_prefix}.wav")
    response = translate([f'{temp_prefix}.wav'])['result'][0]
    return response

class Onlinization:

  def __init__(self, mask=0, lang="cs"):
    self.m = mask
    self.lang = lang

  def sent_tok(self, sent):
    splitter = SentenceSplitter(language=self.lang)
    return splitter.split(sent)

  def css(self, output, window, merge_last_overlap=False):
    match = SequenceMatcher(None, output, window)
    if merge_last_overlap: # if we want to merge at the last LCS if multiple have the same length
        match = match.get_matching_blocks()
        last_longer_than_one = -1
        max_len = 0
        for i, (a, b, l) in enumerate(match):
            if l >= max_len:
                max_len = l
                last_longer_than_one = i
        return match[last_longer_than_one].a, match[last_longer_than_one].b, match[last_longer_than_one].size
    match = match.find_longest_match(0, len(output), 0, len(window))
    return match.a, match.b, match.size

  def split(self, output, window, merge_last_overlap=False): # output: infinite length output, window: current output window
    output = output.split()
    window = window.split()
    if len(output) == 0:
      return [], window, []
    if len(output) > len(window):
      output_right = output[-len(window):]
      id1, id2, l = self.css(output_right, window, merge_last_overlap)
      id1 = id1 + len(output) - len(window)

    else:
      id1, id2, l = self.css(output, window, merge_last_overlap)

    if l == 0:
      return output, window[id2:id2+l], window

    left_prefix = output[:id1]
    common = window[id2:id2+l]
    right_prefix = window[id2+l:]
    return left_prefix, common, right_prefix

def view(output, w_len):
  output = output.split(" ")
  stable_out = output[:-w_len]
  if len(stable_out) < 1:
    return " ".join(output)
  stable_out = " ".join(stable_out)
  stable_out_last_part = sent_tok(stable_out, "cs")[-1]
  view = stable_out_last_part.split(" ") + output[-w_len:]
  view = " ".join(view)
  return view

def sent_tok(output, lang):
  splitter = SentenceSplitter(language=lang)
  sents = splitter.split(output)
  return sents

def log(fileobj, *items):
  print(*items, file=fileobj)

def extended(args):
  init()
  on = Onlinization(args.lang)
  window_len = args.window_len
  logfile = open(args.log+'.log', 'w')
  for ted_talk in sys.stdin:
      bffr = []
      output = ""
      output_line = 0
      filename, start, offset, duration = ted_talk.strip().split()
      start = float(start)
      offset = float(offset)
      duration = float(duration)
      end = offset + duration
      pointer = args.shift
      while pointer + start < end:  # TODO handle last < second
        if pointer > window_len - 1:
          output_line += 1
          common = []
          current_window_out_len = 1
          prefix_pointer = 0
          while len(common) < current_window_out_len * args.ratio:
            current_window_start = start if window_len > pointer else start + pointer - window_len - prefix_pointer
            current_window_end = start + pointer
            log(logfile, output_line, "Input window:", str(current_window_start)+'\t'+str(current_window_end))
            current_window_out = translate_window(current_window_start, current_window_end, filename, args.log)
            current_window_out_len = len(current_window_out.split())
            log(logfile, output_line, "Output window:", current_window_out)
            lft, common, rght = on.split(output, current_window_out, args.merge_last_overlap)
            if len(common) == 0:
              log(logfile, output_line, "No match at i =", prefix_pointer)
            prefix_pointer += 0.1
            if prefix_pointer > args.max_history:
              break
          if len(common) == 0:
            log(logfile, output_line, "This join has no match!")

          output = lft + common + rght
          output = " ".join(lft + common + rght)
          log(logfile, output_line, "|Common|:", len(common), "|current_window_out|:", current_window_out_len, "ratio:", len(common)/current_window_out_len)
          log(logfile, output_line, "Number of left-extensions:", (prefix_pointer))
          log(logfile, output_line, "Output Stream: ", output )

          if args.slt and args.truncate:
            print(view(output, window_len+i))
          if args.slt and not args.truncate:
            print(output)
        pointer += args.shift

      if not args.slt:
        print(output)
      log(logfile, "<EOT>")
  logfile.close()

switch_id = sys.argv.index('STOP')
fairseq_args = [sys.argv[0]] + sys.argv[switch_id + 1: ]
sys.argv = sys.argv[:switch_id]
args = options()
sys.argv = fairseq_args
eval(args.type)(args)

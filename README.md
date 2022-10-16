# audio_segmentation
Code and data for the paper "Don't Discard Fixed-Window Audio Segmentation in Speech-to-Text Translation"

## Motivation

For real-life online applications of speech-to-text translation, it is crucial that end-to-end spoken language translation models perform well on continuous audio, without relying on human-supplied segmentation. Our findings on five different language pairs show that a simple fixed-window audio segmentation can outperform a state-of-the-art automatic segmentation approach. This repository provides our model outputs and code to reproduce our experiments.

## Reproducing Our Results

Our model outputs can be found in the `model_outputs` directory (ending in `*slt`) together with all files needed to rerun SLTev (files ending in `*ost` and `*ostt`).

Note that we made some modifications to SLTev in order to evaluate delay on translations produced with different audio segmentation methods. To reproduce our results, please use [our fork of SLTev](https://github.com/chanberg/SLTev).

After installing SLTev, you can then run the following command to reproduce our results, selecting the `EXPERIMENT` (model type and segmentation type) and `LANGPAIR` (en-de, es-en, fr-en, it-en, pt-en) of interest:

	SLTeval -i EXPERIMENT.LANGPAIR.slt LANGPAIR.ost LANGPAIR.ostt -f slt ref ostt


## Retraining Our Models

We provide the code for preparing the training data for fine-tuning, the scripts for fine-tuning the models and the scripts for translating.

### Resegmenting the Data
First, download the train and test data for [MuST-C](https://ict.fbk.eu/must-c/) (note that we used version 1.0)  or [mTEDx](http://www.openslr.org/100) (note that we used [mtedx_iwslt2021.tgz](https://www.openslr.org/resources/100/mtedx_iwslt2021.tgz) for testing).

You can then use our scripts to resegment the training data for finetuning on prefixes, prefixes + context or windows as described in our paper. The only files needed are the yaml-file with the segmentation information, the file with the source text (transcription) and the file with the target text (translation).

For prefixes:

	python finetune_scripts/resegment_prefixes.py -y YAML -s SRC -t TRG

For context:

	python finetune_scripts/resegment_context.py -y YAML -s SRC -t TRG

For windows:

	python finetune_scripts/resegment_windows.py -y YAML -s SRC -t TRG

The yaml, source and target outputs will be saved with the same file names plus the ending `.prefix`, `.context`, `.window`.

### Preparing the Data

Please follow the steps in the respective fairseq docs to prepare the training data for [MuST-C](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/ende-mustc.md) and [mTEDx](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_text_joint_to_text/docs/iwslt2021.md).

### Finetuning the Models

Download the pretrained checkpoints for the en-de [MuST-C](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_ave_10.pt) and the multilingual [mTEDx](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/checkpoint17.pt) models and all other necessary files from linked in the fairseq docs.

You can then add the necessary paths and use our training scripts as follows:

For en-de:

	bash finetune_scripts/finetune_must-c.sh

For multilingual (es-en, fr-en, it-en and pt-en):

	bash finetune_scripts/finetune_mtedx.sh

### Translate with Different Segmentations

#### Create the Automatic Segmentations
First, adapt the yaml file of the testset of your choice to use the segmentation method of your choice.

For the gold segmentation, you don't need to change anything except for copying the file to the path mentioned below.

For SHAS, please follow the steps described in the [SHAS repo](https://github.com/mt-upc/SHAS). Make sure to use the pSTREAM algorithm if you want to simulate an online setting.

For fixed window segmentation, you can also use the code in the SHAS repo for length-based segmentation described under "Segmentation with other methods".

For the merging windows approach, the segmentation will happen automatically in the translation script, so here you also don't need to do anything.

Save the resulting yaml files in the location of the original file either as `FILE.gold.yaml`, `FILE.shas.yaml` or  `FILE.fixed.yaml`. Do not remove the original files as these will be used by the window merging scripts.

#### Translate the Testsets

##### Gold, SHAS and Fixed Segmentation

For the original, SHAS and fixed segmentation, you need to set the paths to the data directory, the [fairseq](https://github.com/facebookresearch/fairseq) repo and the model directory (with the checkpoint and other files).

If you want to run the translations with biased beam search enabled, you have to uncomment the options in the script and use this [fairseq fork branch](https://github.com/bhaddow/fairseq/tree/biased-beam) instead.

Then you can call the following script for en-de:

	bash translate_scripts/translate_must-c.sh SRCLANG TRGLANG MODELTYPE SEGTYPE PORT

and for es-en, fr-en, it-en and pt-en:

	bash translate_scripts/translate_mtedx.sh SRCLANG TRGLANG MODELTYPE SEGTYPE PORT

where MODELTYPE is either "original", "prefix", "context" or "window" and SEGTYPE is either "gold", "shas" or "fixed". PORT is the port where the translation server should be running.

Finally, before you can evaluate the output with SLTev (see above), you need to create a specific input format with the following command:

	python translate_scripts/postprocess_gold_shas_fixed.py -i OUTFILE.TRGLANG -a OUTFILE.annotation -o FILE.slt

##### Merging Window Approach

For the merging window approach, you also need to set the path variables in the scripts.

Then you can call the following script for en-de:

	bash translate_scripts/merge_windows_must-c.sh SRCLANG TRGLANG MODELTYPE

and for es-en, fr-en, it-en and pt-en:

	bash translate_scripts/merge_windows_mtedx.sh SRCLANG TRGLANG MODELTYPE

where MODELTYPE is either "original", "prefix", "context" or "window".

Finally, before you can evaluate the output with SLTev (see above), you need to create a specific input format with the following command:

	python translate_scripts/postprocess_merged.py -i OUTFILE.log -o OUTFILE.slt

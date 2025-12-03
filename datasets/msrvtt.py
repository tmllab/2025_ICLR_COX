import os
import json
import pandas as pd
from .base_mmmd_dataset import MultiModalDataset


class MSRVTT(MultiModalDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split
        self.keys = None
        super().__init__(*args, **kwargs,)
        self.modalities = ['vision', 'audio', 'text']


    def _load_metadata(self):

        if self.split=='train':

            self.metadatajson_dir = os.path.join(self.metadata_dir, 'train_val_videodatainfo.json')
            with open(self.metadatajson_dir, 'r') as fid:
                metadata = json.load(fid)
            video_df = pd.DataFrame(metadata['videos'])
            sentence_df = pd.DataFrame(metadata['sentences'])
            video_group = video_df.groupby('split')
            train_video = video_group.get_group('train')

            # some videos does not have audio
            audio_list = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.metadata_dir, 'TrainValAudio'))]
            train_video = train_video[train_video['video_id'].isin(audio_list)]

            self.video_id = train_video['video_id'].values
            self.labels = train_video['category'].values
            train_sentence = sentence_df.loc[sentence_df['video_id'].isin(self.video_id)]
            train_video = train_video.sort_values(by='video_id', ascending=True)
            train_sentence = train_sentence.sort_values(by='video_id', ascending=True)

            self.sentence = {str(vid):[] for vid in train_video['video_id']}
            for vid, sen in self.sentence.items():
                sen += list(train_sentence.groupby('video_id').get_group(vid)['caption'].values)

            self.video_filename = 'TrainValVideo'
            self.audio_filename = 'TrainValAudio'

            del video_df
            del sentence_df

        elif self.split=='val':
            
            self.metadatajson_dir = os.path.join(self.metadata_dir, 'train_val_videodatainfo.json')
            with open(self.metadatajson_dir, 'r') as fid:
                metadata = json.load(fid)
            video_df = pd.DataFrame(metadata['videos'])
            sentence_df = pd.DataFrame(metadata['sentences'])
            video_group = video_df.groupby('split')
            val_video = video_group.get_group('validate')

            # some videos does not have audio
            audio_list = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.metadata_dir, 'TrainValAudio'))]
            val_video = val_video[val_video['video_id'].isin(audio_list)]

            self.video_id = val_video['video_id'].values
            self.labels = val_video['category'].values
            val_sentence = sentence_df.loc[sentence_df['video_id'].isin(self.video_id)]
            val_video = val_video.sort_values(by='video_id', ascending=True)
            val_sentence = val_sentence.sort_values(by='video_id', ascending=True)

            self.sentence = {str(vid):[] for vid in val_video['video_id']}
            for vid, sen in self.sentence.items():
                sen += list(val_sentence.groupby('video_id').get_group(vid)['caption'].values)
            
            self.video_filename = 'TrainValVideo'
            self.audio_filename = 'TrainValAudio'

            del video_df
            del sentence_df

        elif self.split=='test':

            self.metadatajson_dir = os.path.join(self.metadata_dir, 'test_videodatainfo.json')
            with open(self.metadatajson_dir, 'r') as fid:
                metadata = json.load(fid)
            video_df = pd.DataFrame(metadata['videos'])
            sentence_df = pd.DataFrame(metadata['sentences'])
            video_group = video_df.groupby('split')
            test_video = video_group.get_group('test')

            # some videos does not have audio
            audio_list = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.metadata_dir, 'TestAudio'))]
            test_video = test_video[test_video['video_id'].isin(audio_list)]

            self.video_id = test_video['video_id'].values
            self.labels = test_video['category'].values
            test_sentence = sentence_df.loc[sentence_df['video_id'].isin(self.video_id)]
            test_video = test_video.sort_values(by='video_id', ascending=True)
            test_sentence = test_sentence.sort_values(by='video_id', ascending=True)

            self.sentence = {str(vid):[] for vid in test_video['video_id']}
            for vid, sen in self.sentence.items():
                sen += list(test_sentence.groupby('video_id').get_group(vid)['caption'].values)
        
            self.video_filename = 'TestVideo'
            self.audio_filename = 'TestAudio'
        
            del video_df
            del sentence_df
            
        self.keys = self.video_id

    def __getitem__(self, index):
        video_path = os.path.join(self.metadata_dir, self.video_filename, self.keys[index]+'.mp4')
        audio_path = os.path.join(self.metadata_dir, self.audio_filename, self.keys[index]+'.wav')
        sentences = self.sentence[self.keys[index]][0]  # Choose the first sentence
        ret = dict()
        ret.update(self._get_video_item(video_path))
        ret.update(self._get_audio_item(audio_path))
        ret.update(self._get_text_item(sentences))
        ret.update({'im_label': self.im_labels[index]})
        ret.update({'oom_label': self.oom_labels[index]})

        return ret





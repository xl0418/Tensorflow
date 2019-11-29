import json
class audio_process():
    def __init__(self):
        self.txt_sentence_list = []
    def audio_extract(self,audio_dir):
        sentence_lot_add = audio_dir
        with open(sentence_lot_add,encoding="utf8", errors='ignore') as inf:
            dict_from_file = json.load(inf)
        personal_txt = dict_from_file['translation']['data']['result']
        txt_length = len(personal_txt)
        for txt_index_length in range(txt_length):
            self.txt_sentence_list.append(personal_txt[txt_index_length]['text'])
        return self.txt_sentence_list

if __name__ == "__main__":
    audio_dir = "c:/Liang/Tensorflow/MMSEGdata/2019-11-04-9663D1F9F61F428C89C1308F7A74ACB7" \
                "/audio11.txt"

    test_audio_process = audio_process()
    txt_p1 = test_audio_process.audio_extract(audio_dir)
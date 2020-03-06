from MMSEG import Mmseg
from Audio_processing import audio_process
import pandas as pd


class score():
    def __init__(self):
        self.worker_segdata = []
        self.criteria_segdata = []


    def employee_data(self,data_dir):
        test_audio_process = audio_process()
        data_worker = test_audio_process.audio_extract(data_dir)
        testmmseg = Mmseg()
        for sen in data_worker:
            if sen == '啊啊啊啊，': pass
            self.worker_segdata.append(list(testmmseg.cut(sen)))
        return self.worker_segdata

    def criteria(self,cri_dir,sheetname):
        df_lh = pd.read_excel(cri_dir,sheet_name=sheetname)
        txt_lh = df_lh[df_lh.keys()[1]].values[1:].tolist()
        testmmseg = Mmseg()
        for sen in txt_lh:
            self.criteria_segdata.append(list(testmmseg.cut(sen)))
        return self.criteria_segdata

    def data_in_criteria(self,data_dir,cri_dir,sheetname):
        emp_data = self.employee_data(data_dir)
        crit_data = self.criteria(cri_dir,sheetname)
        flatten_cri = [item for sublist in crit_data for item in sublist ]



if __name__ == "__main__":
    audio_dir = "c:/Liang/Tensorflow/MMSEGdata/2019-11-04-9663D1F9F61F428C89C1308F7A74ACB7" \
                "/audio11.txt"
    cri1_dir = 'C:/Liang/Googlebox/industry/criteria.xlsx'
    sheet_name='服务耐心度'
    test_score = score()
    seg_p1 = test_score.employee_data(audio_dir)
    lh_data = test_score.criteria(cri1_dir,sheet_name)
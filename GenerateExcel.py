from xlwt import Workbook
import os
import numpy as np
class GenerateData:

    def __init__(self, dir):
        self.directory = dir
        os.mkdir(dir)


    def generate_data(self, mean, var, number_rep,number_sheet):
        sheet_name = ['sheet '+ str(i) for i in range(1,number_sheet+1)]
        output_filename = ['data '+ str(i) for i in range(1,number_rep)]
        for rep in range(1,number_rep+1):
            # Workbook is created
            excel_data = Workbook()

            for no_sheet in range(1,number_sheet):

                # add_sheet is used to create sheet.
                sheet1 = excel_data.add_sheet(sheet_name[no_sheet])

                sheet1.write(1, 0, 'ISBT DEHRADUN')
                sheet1.write(2, 0, 'SHASTRADHARA')
                sheet1.write(3, 0, 'CLEMEN TOWN')
                sheet1.write(4, 0, 'RAJPUR ROAD')
                sheet1.write(5, 0, 'CLOCK TOWER')
                sheet1.write(0, 1, 'ISBT DEHRADUN')
                sheet1.write(0, 2, 'SHASTRADHARA')
                sheet1.write(0, 3, 'CLEMEN TOWN')
                sheet1.write(0, 4, 'RAJPUR ROAD')
                sheet1.write(0, 5, 'CLOCK TOWER')

            excel_data.save('xlwt example.xls')


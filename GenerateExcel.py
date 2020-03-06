from xlwt import Workbook
import os
import numpy as np

class GenerateData:
    # Initialize the directory for storing data files.
    def __init__(self, dir):
        self.directory = dir
        if os.path.isdir(self.directory):
            pass
        else:
            os.mkdir(dir)

    # Generate 'number_rep' excel files that each contains 'number_sheet' sheets with 'no_row' rows and 'no_col' columns.
    # Each cell is filled with a randomly generated value.
    def generate_data(self,mean,var,no_row,no_col, number_rep,number_sheet):
        # Sheet names
        self.sheet_name = ['sheet '+ str(i) for i in range(1,number_sheet+1)]
        # Data file names
        self.output_filename = ['data '+ str(i) for i in range(1,number_rep+1)]
        # Looping through all files
        for rep in range(number_rep):
            # Excel files are created
            excel_data = Workbook()
            # Looping through all sheets of each file
            for no_sheet in range(number_sheet):
                # Creating sheets
                new_sheet = excel_data.add_sheet(self.sheet_name[no_sheet])
                # Filling cells with values
                for row_index in range(no_row):
                    for col_index in range(no_col):
                        generate_value = np.random.normal(mean,var,1)
                        new_sheet.write(row_index, col_index, generate_value[0])

            excel_data.save(self.directory+'\\'+ self.output_filename[rep] +'.xls')



if __name__ == '__main__':
    # Creating the directory
    dir = os.getcwd() + '\\test'
    test_g = GenerateData(dir)
    # Generating 3 data files with 3 sheets each.
    # Filling random values drawn from a normal distribution with the mean 10, the std 5.
    test_g.generate_data(mean=10,var=5,no_row=10,no_col=3,number_rep=3,number_sheet=3)



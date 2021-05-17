from lib import *

class MyDataset(data.Dataset):
    def __init__(self, input1_list, input2_list, output_list,max_person_info, max_output,phase = "train"):
        self.input1_list = input1_list
        self.input2_list = input2_list
        self.output_list = output_list
        self.phase = phase
        self.max_person_info = max_person_info
        self.max_output = max_output

    def __len__(self):
        return len(self.input1_list)

    def __getitem__(self, idx):
        ppg_input = self.input1_list[idx]
        persion_info = self.input2_list[idx]
        output = self.output_list[idx]

        return ppg_input, persion_info, output

    def get_max_output(self):
        return self.max_output

    def get_max_persion_info(self):
        return self.max_person_info



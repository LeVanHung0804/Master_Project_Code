from lib import *
from config import *
from utils import *
from dataset import *
from model import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--model', metavar='MODEL',default=model_name[2],
                    choices=model_name)

args = parser.parse_args()

ppg, person_info, output, max_person_info, max_output = create_data_list(input_file_name, output_file_name)
data_split = split_data_list(ppg, person_info, output, option= args.model)

##################
# ppg_train,ppg_val ,person_info_train, person_info_val = train_test_split(ppg, person_info,test_size=0.3,shuffle=True,random_state=25)
# output_train,output_val, _,_ =  train_test_split(output, output,test_size=0.3,shuffle=True,random_state=25)
# train_data = [ppg_train,person_info_train,output_train]
# val_data = [ppg_val,person_info_val,output_val]
# test_data = [ppg_val, person_info_val, output_val]
##################

train_data = data_split[0]
val_data = data_split[1]
test_data = data_split[2]

plot_output_distribution(train_data[2],max_output,save_output_distribution)
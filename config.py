from lib import *

torch.manual_seed(804)
np.random.seed(804)
random.seed(804)

batch_size = 64
num_epochs = 400

model_name = ["CNN_PPG", "CNN_PPG_VPG", "CNN_PPG_VPG_APG", "CNN_PPG_VPG_APG_info"]

save_path = {model_name[0]: ['./results/weights/'+model_name[0]+"/weights.pth", "./results/figures/"+model_name[0]+"/"],
                    model_name[1]: ['./results/weights/'+model_name[1]+"/weights.pth", "./results/figures/"+model_name[1]+"/"],
                    model_name[2]: ['./results/weights/'+model_name[2]+"/weights.pth", "./results/figures/"+model_name[2]+"/"],
                    model_name[3]: ['./results/weights/'+model_name[3]+"/weights.pth", "./results/figures/"+model_name[3]+"/"]
                }

save_output_distribution = "./results/figures/output_distribution/"

input_file_name = "input1_2000_10_PPG_VPG_APG_2cycles.xlsx"
output_file_name = "output_2000_10_PPG_VPG_APG_2cycles.xlsx"


learning_rate = 0.001


from lib import *
from config import *
from utils import *
from dataset import *
from model import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--model', metavar='MODEL',default=model_name[3],
                    choices=model_name)

def main():
    # parameters
    args = parser.parse_args()

    save_weights_path = save_path[args.model][0]
    save_figures_path = save_path[args.model][1]

    ppg, person_info, output, max_person_info, max_output = create_data_list(input_file_name,output_file_name)
    data_split = split_data_list(ppg, person_info, output, option = args.model)

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

    train_data = MyDataset(train_data[0], train_data[1], train_data[2], max_person_info,max_output)
    val_data = MyDataset(val_data[0], val_data[1], val_data[2], max_person_info, max_output)
    test_data = cMyDataset(test_data[0], test_data[1], test_data[2], max_person_info, max_output)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False,drop_last = True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False,drop_last = True)

    dataloader_dict = {"train":train_dataloader, "val":val_dataloader, "test":test_dataloader}


    # net
    model = create_model(args.model)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

    criterior = nn.MSELoss()

    history = train_model(model,dataloader_dict,criterior,optimizer,num_epochs,save_weights_path)

    plot_ln_curve(history, save_figures_path)

if __name__ == "__main__":

    main()
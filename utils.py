from lib import *
from config import *

def create_data_list(input_file, output_file):
# file_name = input1_2000_10_PPG_VPG_APG_2cycles.xlsx
    data_dir = os.path.join(os.getcwd(), "data")
    input_path = os.path.join(data_dir,input_file)
    output_path = os.path.join(data_dir, output_file)
    print("===========================")
    print("Start reading excel file...")

# read ppg input
    ppg = pd.read_excel(input_path)
    ppg = ppg.iloc[:,:].values
    ppg = np.array(ppg)
    num_data = ppg.shape[0]
    len_data = ppg.shape[1]
    ppg = np.reshape(ppg, (num_data,len_data,1))
    ppg = np.reshape(ppg, (num_data,1,len_data,1))


# read output and second input

    output = pd.read_excel(output_path)
    output = output.iloc[:,0:7].values
    output = np.array(output)
    output[:,6:7] = (2019*(np.ones_like(output[:,6:7]))) - output[:,6:7]

    person_info = output[:, 2:7]
    output = output[:,0:2]

# normalize output and input 2
    max_output = np.max(output, axis=0)
    max_input_2 = np.max(person_info, axis=0)

    person_info = person_info / max_input_2
    person_info = np.reshape(person_info, (person_info.shape[0], 1, person_info.shape[1]))
    output = output/max_output

    ppg = torch.Tensor(ppg)
    person_info = torch.Tensor(person_info)
    output = torch.Tensor(output)

    print("Finish reading excel file!")

    return ppg, person_info, output, max_input_2, max_output

def split_data_list(ppg, person_info, output_list, option = "CNN_PPG_VPG_APG_info", distribution = [0.8,0.1,0.1]):
    len_data = ppg.shape[0]
    a = [i for i in range(len_data)]
    random.shuffle(a)
    len_train_set = int(np.floor(len_data*distribution[0]))
    len_val_set = int(np.floor(len_data*distribution[1]))
    len_test_set = len_data - len_train_set - len_val_set

    ppg_train = ppg[0:len_train_set]
    person_info_train = person_info[0:len_train_set]
    output_train = output_list[0:len_train_set]

    ppg_val = ppg[len_train_set:len_train_set+len_val_set]
    person_info_val = person_info[len_train_set:len_train_set + len_val_set]
    output_val = output_list[len_train_set:len_train_set+len_val_set]

    ppg_test = ppg[len_train_set+len_val_set:len_data]
    person_info_test = person_info[len_train_set+len_val_set:len_data]
    output_test = output_list[len_train_set+len_val_set:len_data]

    if option == model_name[0]:
        return [ppg_train[:,:,0:1000,:], person_info_train[:,:,0:1], output_train], [ppg_val[:,:,0:1000,:], person_info_val[:,:,0:1], output_val], [ppg_test[:,:,0:1000,:], person_info_test[:,:,0:1], output_test]
    if option == model_name[1]:
        return [ppg_train[:,:,0:2000,:], person_info_train[:,:,0:1], output_train], [ppg_val[:,:,0:2000,:], person_info_val[:,:,0:1], output_val], [ppg_test[:,:,0:2000,:], person_info_test[:,:,0:1], output_test]
    if option == model_name[2]:
        return [ppg_train, person_info_train[:,:,0:1], output_train], [ppg_val, person_info_val[:,:,0:1], output_val], [ppg_test, person_info_test[:,:,0:1], output_test]
    if option == model_name[3]:
        return [ppg_train, person_info_train, output_train], [ppg_val, person_info_val, output_val], [ppg_test, person_info_test, output_test]
    return None


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs, save_weight_path):
    print("===========================")
    print("Start training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # save check point
    min_loss = 1000

    history = {"train": [], "val": [], "test": []}

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs))

        # move network to device(GPU/CPU)
        net.to(device)

        torch.backends.cudnn.benchmark = True

        for phase in ["train", "val", "test"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0

            # if (epoch == 0) and (phase == "train"):
            #     continue
            for input1, input2, outputs in tqdm(dataloader_dict[phase]):
                # move inputs, labels to GPU/GPU
                input1 = input1.to(device)
                input2 = input2.to(device)
                outputs = outputs.to(device)

                # set gradient of optimizer to be zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    predict = net(input1, input2)
                    loss = criterior(predict, outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * input1.size(0)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} ".format(phase, epoch_loss))

            if epoch_loss < min_loss and phase == "val":
                min_loss = epoch_loss
                print("Save check point at epoch {}".format(epoch))
                torch.save(net.state_dict(), save_weight_path)

            history[phase].append(epoch_loss)

    # history["train"].insert(0, history["val"][0])

    print(net.state_dict())

    return history


def load_model(net, model_path):
    load_weights = torch.load(model_path, map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)

    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)

    return net

def plot_ln_curve(history, save_path):
    fig = plt.figure()
    plt.plot(history["train"])
    plt.plot(history["val"])
    plt.title("Learning curve")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["training", "validation"], loc = "upper right")
    save_figure_path = save_path + str(num_epochs) + 'train_phase.png'
    fig.savefig(save_figure_path)

def plot_output_distribution(output, max_output, save_path):
    output = output.numpy()
    output = output*max_output

    sbp = output[:,0]
    dbp = output[:,1]

    # histogram of SBP
    max_sbp = np.max(sbp)
    min_sbp = np.min(sbp)
    # max_bin_sbp = int(np.ceil(max_sbp/10)*10)
    # min_bin_sbp = int(np.round(min_sbp/10)*10)

    max_bin_sbp = int(np.ceil(max_sbp))
    min_bin_sbp = int(np.round(min_sbp))

    bins = np.array(range(min_bin_sbp,max_bin_sbp,1))
    fig = plt.figure()
    plt.hist(sbp, bins)
    plt.ylabel("SBP range")
    plt.title("distribution of SBP")
    fig.savefig(save_path+ "hist of SBP.png")

    # histogram of DBP
    max_dbp = np.max(dbp)
    min_dbp = np.min(dbp)
    # max_bin_dbp = int(np.ceil(max_dbp/10)*10)
    # min_bin_dbp = int(np.round(min_dbp/10)*10)

    max_bin_dbp = int(np.ceil(max_dbp))
    min_bin_dbp = int(np.round(min_dbp))

    bins = np.array(range(min_bin_dbp,max_bin_dbp,1))
    fig = plt.figure()
    plt.hist(dbp, bins)
    plt.ylabel("DBP range")
    plt.title("distribution of DBP")
    fig.savefig(save_path+ "hist of DBP.png")

def plot_pre_out_comparision(predict, output, max_output):

    pre = predict.detach().numpy()*max_output
    out = output.detach().numpy()*max_output

    plt.plot(pre[:,0], out[:,0], 'ro')
    plt.plot(pre[:,1], out[:,1], 'go')
    plt.plot([0, 200], [0, 200])
    plt.title('accuracy of SBP and DBP')
    plt.ylabel('predict')
    plt.xlabel('target')
    plt.legend(['test_SBP', 'test_DBP'], loc='upper right')
    plt.show()

def balance_data(input1, input2, output):
    pass

if __name__ == "__main__":
    ppg, person_info, output, _,_ =create_data_list("input1_2000_10_PPG_VPG_APG_2cycles.xlsx","output_2000_10_PPG_VPG_APG_2cycles.xlsx")
    data_split = split_data_list(ppg, person_info, output)
    a = 1
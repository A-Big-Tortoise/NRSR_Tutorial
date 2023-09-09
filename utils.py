import numpy as np
import matplotlib.pyplot as plt
import pickle

# 此代码需要大改，但暂时可以用
def calc_mae(gt, pred):
    return np.mean(abs(np.array(gt) - np.array(pred)))

def plot_2vectors(label, pred, save=False, name=None):
    """lsit1: label, list2: prediction"""

    list1 = label
    list2 = np.array(pred)
    if list2.ndim == 2:
        mae = calc_mae(list1, list2[:, 0])
    else:
        mae = calc_mae(list1, list2)

    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0, np.min(list2), f'MAE={mae}')

    plt.scatter(np.arange(list2.shape[0]), list2[sorted_id], s=1, alpha=0.5, label=f'{name} prediction', color='blue')
    plt.scatter(np.arange(list1.shape[0]), list1[sorted_id], s=1, alpha=0.5, label=f'{name} label', color='red')
    plt.legend(loc='lower right')

    if save:
        if name is None:
            raise ValueError("If save is True, 'name' argument must be provided.")
        plt.savefig(f'{name}.jpg', dpi=300)
        print(f'Saved plot to {name}.jpg')

    plt.show()

    # mylist = Kshapes_template_test
    # filepath = '../Data/NewTry/Kshapes_templates_0.1_3000_test.csv'

    # def write_csv(mylist, filepath):
    #     import csv
    #     with open(filepath, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(mylist)

    # def write_csv(filepath):
    #     import csv
    #     with open(filepath, 'r') as file:
    #         reader = csv.reader(file)
    #         # for row in reader:
    #             # row
    #             # print(row)
    #     return reader

    # write_csv(mylist, filepath)
    # new_kshapes_train = read_csv(filepath)
    # with open(filepath, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(mylist)

    def ls2pkl(filepath, data):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def pkl2ls(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    def dic2pkl(filepath, data):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def dicl2ls(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

def plot_noise_signal(original_signal, noisy_signal, title_name):
    plt.plot(noisy_signal, label='Noisy Signal')
    plt.plot(original_signal, label='Original Signal')
    plt.title(title_name)
    plt.legend()
    plt.show()

def plot_decomposed_components(signal, components, title_name):
    n_components = len(components)

    plt.subplots(n_components+1, 1)
    plt.subplot(n_components+1, 1, 1)
    plt.title(title_name)

    plt.plot(signal, label='Original Signal', color='r')

    for cnt, component in enumerate(components):
        # print(cnt+1, n_components)
        plt.subplot(n_components+1, 1, cnt+2)
        plt.plot(component, label='Component'+str(cnt+1))
        plt.legend()

    plt.show()
import numpy as np

results = []
def moving_average(data):

    for i in range(data.shape[0]+1):
        if i == 0:
            filtered = data[0 : 1][0]
        elif i < 5:
            filtered = np.sum(data[0 : i], axis = 0)/data[0 : i].shape[0]
        else:
            filtered = np.sum(data[i-5 : i], axis = 0)/(data[i-5 : i].shape[0]) # n[start_index, numFrom1st arr]

        results.append(filtered)

    return np.array((results))

def final_input(data):
    final_data = -data[5:]
    np.save('dog_pace.npy', final_data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    raw_data = np.load('raw_data_100.npy')
    filtered = moving_average(raw_data)


    final_input(filtered)



    labe_name = ["right_front_thigh_angle", "right_front_calf_angle", "right_rear_thigh_angle", "right_rear_calf_angle"]

    plt.figure(0)
    plt.plot((filtered), '.-', label=labe_name)
    plt.legend(loc='best')

    # plt.figure(1)
    # plt.plot(np.rad2deg(raw_data), '.-', label=labe_name)
    # plt.legend(loc='best')
    plt.show()

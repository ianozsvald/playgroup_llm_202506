def transform(initial):
    import numpy as np
    initial = np.array(initial)
    assert initial.shape == (3,3)

    # Find the frequency of each value in "initial"
    freq_dict = {}
    for i in range(3):
        for j in range(3):
            if initial[i][j] not in freq_dict:
                freq_dict[initial[i][j]] = 1
            else:
                freq_dict[initial[i][j]] += 1

    # Find the most frequent value, least frequent values, and middle-frequency values
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    most_frequent_value = sorted_freq[0][0]
    #least_frequent_values = [x[0] for x in sorted_freq[-2:]]
    #middle_frequency_values = [x[0] for x in sorted_freq[1:-2]]

    # Replace the least frequent values with a new value
    #new_value = max(freq_dict) + 1
    new_value = 5
    final = initial.copy()
    for i in range(3):
        for j in range(3):
            if final[i][j] != most_frequent_value:
                final[i][j] = new_value

    assert final.shape == (3,3)
    return final

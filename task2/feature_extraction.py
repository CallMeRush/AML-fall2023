import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def getPPositions(ecg_proc=None, show=False):
    template_r_position = 60
    template_p_position_max = 35
    
    P_positions = []
    P_start_positions = []
    P_end_positions = []

    for i, each in enumerate(ecg_proc["templates"]):
        # Get P position
        template_left = each[0 : template_p_position_max + 1]
        max_from_template_left = np.argmax(template_left)
        P_position = ecg_proc["rpeaks"][i] - template_r_position + max_from_template_left
        P_positions.append(P_position)

        # Get P start position
        template_P_left = each[0 : max_from_template_left + 1]
        mininums_from_template_left = argrelextrema(template_P_left, np.less)
        # MY CORRECTION
        if len(mininums_from_template_left[0]) == 0:
            mininums_from_template_left = []
            mininums_from_template_left.append([0])
        # print("P start position=" + str(mininums_from_template_left[0][-1]))
        P_start_position = ecg_proc["rpeaks"][i] - template_r_position + mininums_from_template_left[0][-1]
        P_start_positions.append(P_start_position)

        # Get P end position
        template_P_right = each[max_from_template_left : template_p_position_max + 1]
        mininums_from_template_right = argrelextrema(template_P_right, np.less)
        # MY CORRECTION
        if len(mininums_from_template_right[0]) == 0:
            mininums_from_template_right = []
            mininums_from_template_right.append([np.argmin(template_P_right)])
        # print("P end position=" + str(mininums_from_template_right[0][0]+max_from_template_left))
        P_end_position = ecg_proc["rpeaks"][i] - template_r_position + max_from_template_left + mininums_from_template_right[0][0]
        P_end_positions.append(P_end_position)

        if show:
            plt.plot(each)
            plt.axvline(x=template_r_position, color="r", label="R peak")
            plt.axvline(x=max_from_template_left, color="yellow", label="P Position")
            plt.axvline(x=mininums_from_template_left[0][-1], color="green", label="P start")
            plt.axvline(x=(max_from_template_left + mininums_from_template_right[0][0]), color="green", label="P end")
            plt.legend()
            plt.show()
    return np.array(P_positions), np.array(P_start_positions), np.array(P_end_positions)

def getQPositions(ecg_proc=None, show=False):
    template_r_position = 60  # R peek on the template is always on 100 index
    Q_positions = []
    Q_start_positions = []

    for i, each in enumerate(ecg_proc["templates"]):
        # Get Q Position
        template_left = each[0 : template_r_position + 1]
        mininums_from_template_left = argrelextrema(template_left, np.less)
        # MY CORRECTION
        if len(mininums_from_template_left[0]) == 0:
            mininums_from_template_left = []
            mininums_from_template_left.append([np.argmin(template_left)])
        # print("Q position= " + str(mininums_from_template_left[0][-1]))
        Q_position = ecg_proc["rpeaks"][i] - template_r_position + mininums_from_template_left[0][-1]
        #Q_position = ecg_proc["rpeaks"][n] - template_r_position + np.argmin(template_left)
        Q_positions.append(Q_position)

        # Get Q start position
        template_Q_left = each[0 : np.argmin(template_left) + 1]
        maximum_from_template_Q_left = argrelextrema(template_Q_left, np.greater)
        # MY CORRECTION
        if len(maximum_from_template_Q_left[0]) == 0:
            maximum_from_template_Q_left = []
            maximum_from_template_Q_left.append([np.argmax(template_Q_left)])
        # print("Q start position=" + str(maximum_from_template_Q_left[0][-1]))
        # print("Q start value=" + str(template_Q_left[maximum_from_template_Q_left[0][-1]]))
        Q_start_position = ecg_proc["rpeaks"][i] - template_r_position + maximum_from_template_Q_left[0][-1]
        Q_start_positions.append(Q_start_position)

        if show:
            plt.plot(each)
            plt.axvline(x=template_r_position, color="r", label="R peak")
            plt.axvline(x=mininums_from_template_left[0][-1], color="yellow", label="Q Position")
            plt.axvline(x=maximum_from_template_Q_left[0][-1], color="green", label="Q Start Position")
            plt.legend()
            plt.show()
    return np.array(Q_positions), np.array(Q_start_positions)

def getSPositions(ecg_proc=None, show=False):
    template_r_position = 60  # R peek on the template is always on 100 index
    S_positions = []
    S_end_positions = []
    template_size = len(ecg_proc["templates"][0])

    for i, each in enumerate(ecg_proc["templates"]):
        # Get S Position
        template_right = each[template_r_position : template_size + 1]
        mininums_from_template_right = argrelextrema(template_right, np.less)
        S_position = ecg_proc["rpeaks"][i] + mininums_from_template_right[0][0]
        S_positions.append(S_position)

        # Get S end position
        maximums_from_template_right = argrelextrema(template_right, np.greater)
        # MY CORRECTION
        if len(maximums_from_template_right[0]) == 0:
            maximums_from_template_right = []
            maximums_from_template_right.append([np.argmax(template_right)])
        # print("S end position=" + str(maximums_from_template_right[0][0]))
        # print("S end value=" + str(template_right[maximums_from_template_right[0][0]]))
        S_end_position = ecg_proc["rpeaks"][i] + maximums_from_template_right[0][0]
        S_end_positions.append(S_end_position)

        if show:
            plt.plot(each)
            plt.axvline(x=template_r_position, color="r", label="R peak")
            plt.axvline(x=template_r_position + mininums_from_template_right[0][0], color="yellow", label="S Position")
            plt.axvline(x=template_r_position + maximums_from_template_right[0][0], color="green", label="S end Position")
            plt.legend()
            plt.show()
    return np.array(S_positions), np.array(S_end_positions)

def getTPositions(ecg_proc=None, show=False):
    template_r_position = 60  # R peek on the template is always on 100 index
    template_T_position_min = 100  # the T will be always hapenning after 150 indexes of the template
    T_positions = []
    T_start_positions = []
    T_end_positions = []

    for i, each in enumerate(ecg_proc["templates"]):
        # Get T position
        template_right = each[template_T_position_min:]
        max_from_template_right = np.argmax(template_right)
        # print("T Position=" + str(template_T_position_min + max_from_template_right))
        T_position = ecg_proc["rpeaks"][i] - template_r_position + template_T_position_min + max_from_template_right
        T_positions.append(T_position)

        # Get T start position
        template_T_left = each[template_r_position : template_T_position_min + max_from_template_right]
        min_from_template_T_left = argrelextrema(template_T_left, np.less)
        # MY CORRECTION
        if len(min_from_template_T_left[0]) == 0:
            min_from_template_T_left = []
            min_from_template_T_left.append([np.argmin(template_T_left)])
        # print("T start position=" + str(template_r_position+min_from_template_T_left[0][-1]))
        T_start_position = ecg_proc["rpeaks"][i] + min_from_template_T_left[0][-1]
        T_start_positions.append(T_start_position)

        # Get T end position
        template_T_right = each[template_T_position_min + max_from_template_right :]
        mininums_from_template_T_right = argrelextrema(template_T_right, np.less)
        # MY CORRECTION
        if len(mininums_from_template_T_right[0]) == 0:
            mininums_from_template_T_right = []
            mininums_from_template_T_right.append([np.argmin(template_T_right)])
        # print("T end position=" + str(template_T_position_min + max_from_template_right + mininums_from_template_T_right[0][0]))
        T_end_position = ecg_proc["rpeaks"][i] - template_r_position + template_T_position_min + max_from_template_right + mininums_from_template_T_right[0][0]
        T_end_positions.append(T_end_position)

        if show:
            plt.plot(each)
            plt.axvline(x=template_r_position, color="r", label="R peak")
            plt.axvline(x=template_T_position_min + max_from_template_right, color="yellow", label="T Position")
            plt.axvline(x=template_r_position + min_from_template_T_left[0][-1], color="green", label="P start")
            plt.axvline(x=(template_T_position_min + max_from_template_right + mininums_from_template_T_right[0][0]), color="green", label="P end")
            plt.legend()
            plt.show()
    return np.array(T_positions), np.array(T_start_positions), np.array(T_end_positions)

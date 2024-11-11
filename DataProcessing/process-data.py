import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import math
import xlsxwriter
from sklearn import preprocessing
from scipy.interpolate import interp1d
import os


############################### ACRONYMS ###############################

#    ACT1 :: Activity of lifting the leg
#    ACT2 :: Activity of arising from a chair
#    ACT31 :: Activity of extending the arms forward for 10 seconds
#    ACT32 :: Activity of touching the nose

########################################################################


############################### SEGMENTATION ###############################

def segment_data_by_indexes(data, segment_indexes, reprocess: bool):
    """
    Segments the input data based on the provided indexes.

    Args:
        data (pd.DataFrame): The input data to be segmented.
        segment_indexes (list[tuple[int, int]] / tuple[int, int]): A list of tuples, or a single tuple,     
            where each tuple contains the start and end indexes to slice the data.
        reprocess (bool): A flag indicating whether each segment's first sample should be relocated to zero.

    Returns:
        list[list[pd.DataFrame]]: A list of data segments.
    """
    wave_segments = []
    # If a single tuple is passed (ACT31), treat it as the full range of data (no actual segmentation).
    if isinstance(segment_indexes, tuple):
        segment_indexes = [segment_indexes]

    for start, end in segment_indexes:
        if isinstance(data, np.ndarray):
            wave_segment = data[start:end]
        else:
            wave_segment = data.iloc[start:end]
        # If ACT2 (mediolateral COM), relocate each segment to start at zero.
        if reprocess:
            first_sample = wave_segment.iloc[0]
            wave_segment -= first_sample

        wave_segments.append(wave_segment)
    return wave_segments

def segment_activity(data, participant: int, activity: int, activity_indexes):
    """
    Segments data based on the given participant, activity, and segments' indexes.

    Args:
        data (pd.DataFrame): The data to segment, each column corresponding to parameter data.
        participant (int): The participant index (1-based).
        activity (int): The activity identifier (1, 2, 31, or 32).
        activity_indexes (list): A list of lists of tuples (for ACT1, ACT2 and ACT32), or just a list of 
            tuples (for ACT31), with each element corresponding to each participant.
    
    Returns:
        list: A list containing lists of segmented data for all parameters
        int: The maximum segment length.
    """
    segments_all_parameters = []  # List of segmented data for all parameters
    participant_indexes = activity_indexes[participant - 1]  # Participant-specific segment indexes
    
    parameters = {
        1: activity1_headers,
        2: activity2_headers,
        31: activity31_headers,
        32: activity32_headers
    }.get(activity, [])

    for parameter in parameters:
        array_data = data[parameter]

        # Apply second-order high-pass filter to the Position parameters of ACT32 and ACT31.
        if (activity == 32 and "Position" in parameter) or activity == 31:
            array_data = apply_high_pass_filter_second_order(array_data)

        # Segment data for ACT1 or ACT32, according to left and right sides.
        if activity == 1 or activity == 32:
            for side, index in zip(["Right", "Left"], participant_indexes):
                if side in parameter:
                    parameter_segments = segment_data_by_indexes(array_data, index, False)
                    segments_all_parameters.append(parameter_segments)
        
        # Segment data for ACT2 (relocating mediolateral COM segments to start at 0).
        elif activity == 2:
            reprocess = "COM x" in parameter
            parameter_segments = segment_data_by_indexes(array_data, participant_indexes, reprocess)
            segments_all_parameters.append(parameter_segments)

        # Segment data for ACT31 (already filtered).
        elif activity == 31:
            parameter_segments = segment_data_by_indexes(array_data, participant_indexes, False)
            segments_all_parameters.append(parameter_segments)

    # Calculate the maximum segment length for further interpolation.
    max_length = max(len(segment) for sublist in segments_all_parameters for segment in sublist)

    return segments_all_parameters, max_length

############################### MEAN CURVES ###############################

def convert_to_array(activity_segments: list, height: float):
    """
    Normalizes the data segments by dividing each value by the participant's height.

    Args:
        activity_segments (list): A list of segments for each parameter.
        height (float): The participant's height used for normalization.
    
    Returns:
        list: A list of normalized segments for each parameter.
    """
    segments_by_parameter = []
    for parameter in activity_segments:
        segments_by_parameter.append([[value / height for value in segment] for segment in parameter])
        
    return segments_by_parameter

def calculate_mean_curve(activity_segments: list, max_length: int):
    """
    Calculates the mean and standard deviation curves from all segmented data for each parameter.

    Args:
        activity_segments (list): Segmented data for each parameter.
        max_length (int): The max length for interpolated curves.

    Returns:
        list: contains the mean curves
        list: contains the standard deviation curves.
    """
    interp_range = np.linspace(0, 1, max_length) # Precompute interpolation range
    
    mean_curves = []
    std_dev_curves = []

    for parameter_segments in activity_segments:
        # Interpolate each curve in the parameter segments.
        interpolated_curves = [
            interp1d(np.linspace(0, 1, len(curve)), curve)(interp_range)
            for curve in parameter_segments
            if curve is not None and isinstance(curve, (list, np.ndarray)) and len(curve) > 0
        ]
        # Convert to array and compute mean and standard deviation, processing only if there are valid interpolated curves.
        if interpolated_curves:
            curves_array = np.array(interpolated_curves)
            mean_curves.append(np.mean(curves_array, axis=0))
            std_dev_curves.append(np.std(curves_array, axis=0))
        else:
            # Handle cases with no valid curves.
            mean_curves.append(np.zeros(max_length))
            std_dev_curves.append(np.zeros(max_length))

    return mean_curves, std_dev_curves

############################### PLOT DATA ###############################

def plot_data(data, participant: int, activity: int, session: str, type: str, healthy: bool, process: str):
    """
    Plots the given data for a participant based on activity, process (segmentation or mean curve), and health status.

    Args:
        data (list): The data to plot, either mean curves or segmented curves.
        participant (int): The participant's identifier (1-indexed).
        activity (int): The activity type identifier.
        session (str): The session identifier (A, B or C).
        type (str): Evaluation time point (Pre, Post or During).
        healthy (bool): Indicates if the participant is healthy (True) or has Parkinson's (False).
        process (str): Type of process (Segmentation or Mean Curve).
    """
    activity_params = {
        1: (2, 2, activity1_headers),
        2: (1, 2, activity2_headers),
        31: (2, 3, activity31_headers),
        32: (2, 2, activity32_headers),
    }
    rows, cols, headers = activity_params.get(activity, (2, 2, []))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    for j, parameter in enumerate(data):
        current_ax = axes.flatten()[j]
        if process == "Segmentation":
            for curve in parameter:
                current_ax.plot(curve, '.')
        else:
            current_ax.plot(parameter, '.')
        current_ax.set_title(f"{headers[j]}")
        current_ax.legend()

    suptitle = f"{process} - Participant {participant} - Session {session} - Activity {activity}-{type}"
    folder = "Healthy" if healthy else "Parkinson"
    participant_id = participant if healthy else participant + 9
    savefig = f"{folder}/participant_{participant_id}/session_{session.lower()}/Activity{activity}/plots/{process.lower()}_act_{activity}{type}.png"

    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6)
    plt.suptitle(suptitle)
    plt.savefig(savefig)
    #plt.show()

def plot_final_data_healthy(mean_curves_A_PRE, std_dev_curves_A_PRE, mean_curves_A_POS, std_dev_curves_A_POS, mean_curves_B_PRE, std_dev_curves_B_PRE, mean_curves_B_POS, std_dev_curves_B_POS, mean_curves_C_PRE, std_dev_curves_C_PRE, mean_curves_C_DURING, std_dev_curves_C_DURING, mean_curves_C_POS, std_dev_curves_C_POS, participant: int, activity: int, all_participants: bool):
    if activity == 1 or activity == 32:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(65, 48))
    elif activity == 2:
        fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(35, 12))
    else:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(6, 3, figsize=(75, 90))

    body_position = ""
    body_velocity = ""
    if activity == 1:
        body_position = "Foot Z"
        body_velocity = "Foot Z"
        data_reference_right_pos = ACT1_RIGHT_FOOT_POS
        data_reference_left_pos = ACT1_LEFT_FOOT_POS 
        data_reference_right_vel = ACT1_RIGHT_FOOT_VEL
        data_reference_left_vel = ACT1_LEFT_FOOT_VEL
        length_A = 79
        length_B = 88
        length_C = 108
    if activity == 32:
        body_position = "Hand Z"
        body_velocity = "Hand Y"
        data_reference_right_pos = ACT32_RIGHT_HAND_POS
        data_reference_left_pos = ACT32_LEFT_HAND_POS 
        data_reference_right_vel = ACT32_RIGHT_HAND_VEL
        data_reference_left_vel = ACT32_LEFT_HAND_VEL
        length_A = 874
        length_B = 915
        length_C = 672

    handles, labels = [], []

    if activity == 1 or activity == 32:
        # ----------------- POSITION -----------------
        max_value = max(np.max(mean_curves_A_PRE[0]), np.max(mean_curves_A_POS[0]), np.max(mean_curves_A_PRE[1]), np.max(mean_curves_A_POS[1]), np.max(mean_curves_B_PRE[0]), np.max(mean_curves_B_POS[0]), np.max(mean_curves_B_PRE[1]), np.max(mean_curves_B_POS[1]), np.max(mean_curves_C_PRE[0]), np.max(mean_curves_C_POS[0]), np.max(mean_curves_C_PRE[1]), np.max(mean_curves_C_POS[1]))
        min_value = min(np.min(mean_curves_A_PRE[0]), np.min(mean_curves_A_POS[0]), np.min(mean_curves_A_PRE[1]), np.min(mean_curves_A_POS[1]), np.min(mean_curves_B_PRE[0]), np.min(mean_curves_B_POS[0]), np.min(mean_curves_B_PRE[1]), np.min(mean_curves_B_POS[1]), np.min(mean_curves_C_PRE[0]), np.min(mean_curves_C_POS[0]), np.min(mean_curves_C_PRE[1]), np.min(mean_curves_C_POS[1]))

        ax1.plot(np.arange(len(mean_curves_A_PRE[0])), mean_curves_A_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax1.plot(np.arange(len(mean_curves_A_POS[0])), mean_curves_A_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 32:
            ax1.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax1.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower0 = np.array(mean_curves_A_PRE[0]) - np.array(std_dev_curves_A_PRE[0])
        upper0 = np.array(mean_curves_A_PRE[0]) + np.array(std_dev_curves_A_PRE[0])
        ax1.fill_between(np.arange(len(mean_curves_A_PRE[0])), lower0, upper0, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower1 = np.array(mean_curves_A_POS[0]) - np.array(std_dev_curves_A_POS[0])
        upper1 = np.array(mean_curves_A_POS[0]) + np.array(std_dev_curves_A_POS[0])
        ax1.fill_between(np.arange(len(mean_curves_A_POS[0])), lower1, upper1, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.53), label='Std. Post-VRMT')

        ax1.set_ylabel('Position', fontsize=65, labelpad=22)
        ax1.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax1.text(0.5, 1.085,  f'Right {body_position}', fontsize=65, ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.22, 'Session A', fontsize=65, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=65)
        ax1.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax1.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax1.set_ylim([min_value-0.0015, max_value+0.0015])

        ax2.plot(np.arange(len(mean_curves_B_PRE[0])), mean_curves_B_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax2.plot(np.arange(len(mean_curves_B_POS[0])), mean_curves_B_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 1:
            ax2.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax2.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower4 = np.array(mean_curves_B_PRE[0]) - np.array(std_dev_curves_B_PRE[0])
        upper4 = np.array(mean_curves_B_PRE[0]) + np.array(std_dev_curves_B_PRE[0])
        ax2.fill_between(np.arange(len(mean_curves_B_PRE[0])), lower4, upper4, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower5 = np.array(mean_curves_B_POS[0]) - np.array(std_dev_curves_B_POS[0])
        upper5 = np.array(mean_curves_B_POS[0]) + np.array(std_dev_curves_B_POS[0])
        ax2.fill_between(np.arange(len(mean_curves_B_POS[0])), lower5, upper5, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.53), label='Std. Post-VRMT')

        ax2.set_ylabel('Position', fontsize=65, labelpad=22)
        ax2.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax2.text(0.5, 1.085,  f'Right {body_position}', fontsize=65, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 1.22, 'Session B', fontsize=65, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', labelsize=65)
        ax2.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax2.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax2.set_ylim([min_value-0.0015, max_value+0.0015])

        ax3.plot(np.arange(len(mean_curves_C_PRE[0])), mean_curves_C_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        #ax3.plot(np.arange(len(mean_curves_C_DURING[0])), mean_curves_C_DURING[0], '-', label=f'Right During-ME', color=plt.cm.Purples(1.0), linewidth=4.0)
        ax3.plot(np.arange(len(mean_curves_C_POS[0])), mean_curves_C_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 1:
            ax3.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax3.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower8 = np.array(mean_curves_C_PRE[0]) - np.array(std_dev_curves_C_PRE[0])
        upper8 = np.array(mean_curves_C_PRE[0]) + np.array(std_dev_curves_C_PRE[0])
        ax3.fill_between(np.arange(len(mean_curves_C_PRE[0])), lower8, upper8, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        #lower10 = np.array(mean_curves_C_DURING[0]) - np.array(std_dev_curves_C_DURING[0])
        #upper10 = np.array(mean_curves_C_DURING[0]) + np.array(std_dev_curves_C_DURING[0])
        #ax3.fill_between(np.arange(len(mean_curves_C_DURING[0])), lower10, upper10, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.53), label='Std. Right During-ME')
        lower9 = np.array(mean_curves_C_POS[0]) - np.array(std_dev_curves_C_POS[0])
        upper9 = np.array(mean_curves_C_POS[0]) + np.array(std_dev_curves_C_POS[0])
        ax3.fill_between(np.arange(len(mean_curves_C_POS[0])), lower9, upper9, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax3.set_ylabel('Position', fontsize=65, labelpad=22)
        ax3.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax3.text(0.5, 1.085,  f'Right {body_position}', fontsize=65, ha='center', va='center', transform=ax3.transAxes)
        ax3.text(0.5, 1.22, 'Session C', fontsize=65, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True)
        ax3.tick_params(axis='x', labelsize=65)
        ax3.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax3.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax3.set_ylim([min_value-0.0015, max_value+0.0015])

        ax4.plot(np.arange(len(mean_curves_A_PRE[1])), mean_curves_A_PRE[1], '-', label='Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax4.plot(np.arange(len(mean_curves_A_POS[1])), mean_curves_A_POS[1], '-', label='Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 1:
            ax4.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax4.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower2 = np.array(mean_curves_A_PRE[1]) - np.array(std_dev_curves_A_PRE[1])
        upper2 = np.array(mean_curves_A_PRE[1]) + np.array(std_dev_curves_A_PRE[1])
        ax4.fill_between(np.arange(len(mean_curves_A_PRE[1])), lower2, upper2, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower3 = np.array(mean_curves_A_POS[1]) - np.array(std_dev_curves_A_POS[1])
        upper3 = np.array(mean_curves_A_POS[1]) + np.array(std_dev_curves_A_POS[1])
        ax4.fill_between(np.arange(len(mean_curves_A_POS[1])), lower3, upper3, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax4.set_ylabel('Position', fontsize=65, labelpad=22)
        ax4.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax4.text(0.5, 1.085,  f'Left {body_position}', fontsize=65, ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True)
        ax4.tick_params(axis='x', labelsize=65)
        ax4.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax4.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax4.set_ylim([min_value-0.0015, max_value+0.0015])

        ax5.plot(np.arange(len(mean_curves_B_PRE[1])), mean_curves_B_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax5.plot(np.arange(len(mean_curves_B_POS[1])), mean_curves_B_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 1:  
            ax5.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax5.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower6 = np.array(mean_curves_B_PRE[1]) - np.array(std_dev_curves_B_PRE[1])
        upper6 = np.array(mean_curves_B_PRE[1]) + np.array(std_dev_curves_B_PRE[1])
        ax5.fill_between(np.arange(len(mean_curves_B_PRE[1])), lower6, upper6, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower7 = np.array(mean_curves_B_POS[1]) - np.array(std_dev_curves_B_POS[1])
        upper7 = np.array(mean_curves_B_POS[1]) + np.array(std_dev_curves_B_POS[1])
        ax5.fill_between(np.arange(len(mean_curves_B_POS[1])), lower7, upper7, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax5.set_ylabel('Position', fontsize=65, labelpad=22)
        ax5.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax5.text(0.5, 1.085,  f'Left {body_position}', fontsize=65, ha='center', va='center', transform=ax5.transAxes)
        ax5.grid(True)
        ax5.tick_params(axis='x', labelsize=65)
        ax5.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax5.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax5.set_ylim([min_value-0.0015, max_value+0.0015])

        ax6.plot(np.arange(len(mean_curves_C_PRE[1])), mean_curves_C_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax6.plot(np.arange(len(mean_curves_C_POS[1])), mean_curves_C_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        if activity == 1:
            ax6.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)
        else:
            ax6.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower11 = np.array(mean_curves_C_PRE[1]) - np.array(std_dev_curves_C_PRE[1])
        upper11 = np.array(mean_curves_C_PRE[1]) + np.array(std_dev_curves_C_PRE[1])
        ax6.fill_between(np.arange(len(mean_curves_C_PRE[1])), lower11, upper11, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower12 = np.array(mean_curves_C_POS[1]) - np.array(std_dev_curves_C_POS[1])
        upper12 = np.array(mean_curves_C_POS[1]) + np.array(std_dev_curves_C_POS[1])
        ax6.fill_between(np.arange(len(mean_curves_C_POS[1])), lower12, upper12, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax6.set_ylabel('Position', fontsize=65, labelpad=22)
        ax6.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax6.text(0.5, 1.085,  f'Left {body_position}', fontsize=65, ha='center', va='center', transform=ax6.transAxes)
        ax6.grid(True)
        ax6.tick_params(axis='x', labelsize=65)
        ax6.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax6.set_ylim([min_value-0.02, max_value+0.06])
        else:
            ax6.set_ylim([min_value-0.0015, max_value+0.0015])

        # ----------------- VELOCITY -----------------

        max_value = max(np.max(mean_curves_A_PRE[2]), np.max(mean_curves_A_POS[2]), np.max(mean_curves_A_PRE[3]), np.max(mean_curves_A_POS[3]), np.max(mean_curves_B_PRE[2]), np.max(mean_curves_B_POS[2]), np.max(mean_curves_B_PRE[3]), np.max(mean_curves_B_POS[3]), np.max(mean_curves_C_PRE[2]), np.max(mean_curves_C_POS[2]), np.max(mean_curves_C_PRE[3]), np.max(mean_curves_C_POS[3]))
        min_value = min(np.min(mean_curves_A_PRE[2]), np.min(mean_curves_A_POS[2]), np.min(mean_curves_A_PRE[3]), np.min(mean_curves_A_POS[3]), np.min(mean_curves_B_PRE[2]), np.min(mean_curves_B_POS[2]), np.min(mean_curves_B_PRE[3]), np.min(mean_curves_B_POS[3]), np.min(mean_curves_C_PRE[2]), np.min(mean_curves_C_POS[2]), np.min(mean_curves_C_PRE[3]), np.min(mean_curves_C_POS[3]))

        ax7.plot(np.arange(len(mean_curves_A_PRE[2])), mean_curves_A_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax7.plot(np.arange(len(mean_curves_A_POS[2])), mean_curves_A_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        ax7.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_right_vel)), data_reference_right_vel)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower14 = np.array(mean_curves_A_PRE[2]) - np.array(std_dev_curves_A_PRE[2])
        upper14 = np.array(mean_curves_A_PRE[2]) + np.array(std_dev_curves_A_PRE[2])
        ax7.fill_between(np.arange(len(mean_curves_A_PRE[2])), lower14, upper14, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower15 = np.array(mean_curves_A_POS[2]) - np.array(std_dev_curves_A_POS[2])
        upper15 = np.array(mean_curves_A_POS[2]) + np.array(std_dev_curves_A_POS[2])
        ax7.fill_between(np.arange(len(mean_curves_A_POS[2])), lower15, upper15, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.53), label='Std. Post-VRMT')

        ax7.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax7.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax7.text(0.5, 1.085,  f'Right {body_velocity}', fontsize=65, ha='center', va='center', transform=ax7.transAxes)
        ax7.grid(True)
        ax7.tick_params(axis='x', labelsize=65)
        ax7.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax7.set_ylim([min_value-0.35, max_value+0.4])
        else:
            ax7.set_ylim([min_value-0.18, max_value+0.19])

        ax8.plot(np.arange(len(mean_curves_B_PRE[2])), mean_curves_B_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax8.plot(np.arange(len(mean_curves_B_POS[2])), mean_curves_B_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        ax8.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_right_vel)), data_reference_right_vel)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower18 = np.array(mean_curves_B_PRE[2]) - np.array(std_dev_curves_B_PRE[2])
        upper18 = np.array(mean_curves_B_PRE[2]) + np.array(std_dev_curves_B_PRE[2])
        ax8.fill_between(np.arange(len(mean_curves_B_PRE[2])), lower18, upper18, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower19 = np.array(mean_curves_B_POS[2]) - np.array(std_dev_curves_B_POS[2])
        upper19 = np.array(mean_curves_B_POS[2]) + np.array(std_dev_curves_B_POS[2])
        ax8.fill_between(np.arange(len(mean_curves_B_POS[2])), lower19, upper19, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.53), label='Std. Post-VRMT')

        ax8.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax8.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax8.text(0.5, 1.085,  f'Right {body_velocity}', fontsize=65, ha='center', va='center', transform=ax8.transAxes)
        ax8.grid(True)
        ax8.tick_params(axis='x', labelsize=65)
        ax8.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax8.set_ylim([min_value-0.35, max_value+0.4])
        else:
            ax8.set_ylim([min_value-0.18, max_value+0.19])

        ax9.plot(np.arange(len(mean_curves_C_PRE[2])), mean_curves_C_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        #ax6.plot(np.arange(len(mean_curves_C_DURING[2])), mean_curves_C_DURING[2], '-', label=f'Right During-ME', color=plt.cm.Purples(1.0), linewidth=4.0)
        ax9.plot(np.arange(len(mean_curves_C_POS[2])), mean_curves_C_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        ax9.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_right_vel)), data_reference_right_vel)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower22 = np.array(mean_curves_C_PRE[2]) - np.array(std_dev_curves_C_PRE[2])
        upper22 = np.array(mean_curves_C_PRE[2]) + np.array(std_dev_curves_C_PRE[2])
        ax9.fill_between(np.arange(len(mean_curves_C_PRE[2])), lower22, upper22, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        #lower24 = np.array(mean_curves_C_DURING[2]) - np.array(std_dev_curves_C_DURING[2])
        #upper24 = np.array(mean_curves_C_DURING[2]) + np.array(std_dev_curves_C_DURING[2])
        #ax6.fill_between(np.arange(len(mean_curves_C_DURING[2])), lower24, upper24, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.53), label='Std. Right During-ME')
        lower23 = np.array(mean_curves_C_POS[2]) - np.array(std_dev_curves_C_POS[2])
        upper23 = np.array(mean_curves_C_POS[2]) + np.array(std_dev_curves_C_POS[2])
        ax9.fill_between(np.arange(len(mean_curves_C_POS[2])), lower23, upper23, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax9.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax9.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax9.text(0.5, 1.085,  f'Right {body_velocity}', fontsize=65, ha='center', va='center', transform=ax9.transAxes)
        ax9.grid(True)
        ax9.tick_params(axis='x', labelsize=65)
        ax9.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax9.set_ylim([min_value-0.35, max_value+0.4])
        else:
            ax9.set_ylim([min_value-0.18, max_value+0.19])

        ax10.plot(np.arange(len(mean_curves_A_PRE[3])), mean_curves_A_PRE[3], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax10.plot(np.arange(len(mean_curves_A_POS[3])), mean_curves_A_POS[3], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        ax10.plot(np.arange(length_A), interp1d(np.linspace(0, 1, len(data_reference_left_vel)), data_reference_left_vel)(np.linspace(0, 1, length_A)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower16 = np.array(mean_curves_A_PRE[3]) - np.array(std_dev_curves_A_PRE[3])
        upper16 = np.array(mean_curves_A_PRE[3]) + np.array(std_dev_curves_A_PRE[3])
        ax10.fill_between(np.arange(len(mean_curves_A_PRE[3])), lower16, upper16, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower17 = np.array(mean_curves_A_POS[3]) - np.array(std_dev_curves_A_POS[3])
        upper17 = np.array(mean_curves_A_POS[3]) + np.array(std_dev_curves_A_POS[3])
        ax10.fill_between(np.arange(len(mean_curves_A_POS[3])), lower17, upper17, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax10.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax10.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax10.text(0.5, 1.085,  f'Left {body_velocity}', fontsize=65, ha='center', va='center', transform=ax10.transAxes)
        ax10.grid(True)
        ax10.tick_params(axis='x', labelsize=65)
        ax10.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax10.set_ylim([min_value-0.35, max_value+0.35])
        else:
            ax10.set_ylim([min_value-0.18, max_value+0.19])

        ax11.plot(np.arange(len(mean_curves_B_PRE[3])), mean_curves_B_PRE[3], '-', label='Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax11.plot(np.arange(len(mean_curves_B_POS[3])), mean_curves_B_POS[3], '-', label='Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)
        ax11.plot(np.arange(length_B), interp1d(np.linspace(0, 1, len(data_reference_left_vel)), data_reference_left_vel)(np.linspace(0, 1, length_B)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        lower20 = np.array(mean_curves_B_PRE[3]) - np.array(std_dev_curves_B_PRE[3])
        upper20 = np.array(mean_curves_B_PRE[3]) + np.array(std_dev_curves_B_PRE[3])
        ax11.fill_between(np.arange(len(mean_curves_B_PRE[3])), lower20, upper20, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower21 = np.array(mean_curves_B_POS[3]) - np.array(std_dev_curves_B_POS[3])
        upper21 = np.array(mean_curves_B_POS[3]) + np.array(std_dev_curves_B_POS[3])
        ax11.fill_between(np.arange(len(mean_curves_B_POS[3])), lower21, upper21, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax11.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax11.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax11.text(0.5, 1.085,  f'Left {body_velocity}', fontsize=65, ha='center', va='center', transform=ax11.transAxes)
        ax11.grid(True)
        ax11.tick_params(axis='x', labelsize=65)
        ax11.tick_params(axis='y', labelsize=65)
        if activity == 1:
            ax11.set_ylim([min_value-0.35, max_value+0.35])
        else:
            ax11.set_ylim([min_value-0.18, max_value+0.19])

        ax12.plot(np.arange(len(mean_curves_C_PRE[3])), mean_curves_C_PRE[3], '-', label='Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=5.0)
        ax12.plot(np.arange(len(mean_curves_C_POS[3])), mean_curves_C_POS[3], '-', label='Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=5.0)

        lower25 = np.array(mean_curves_C_PRE[3]) - np.array(std_dev_curves_C_PRE[3])
        upper25 = np.array(mean_curves_C_PRE[3]) + np.array(std_dev_curves_C_PRE[3])
        ax12.fill_between(np.arange(len(mean_curves_C_PRE[3])), lower25, upper25, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.53), label='Std. Pre-VRMT')
        lower26 = np.array(mean_curves_C_POS[3]) - np.array(std_dev_curves_C_POS[3])
        upper26 = np.array(mean_curves_C_POS[3]) + np.array(std_dev_curves_C_POS[3])
        ax12.fill_between(np.arange(len(mean_curves_C_POS[3])), lower26, upper26, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.53), label='Std. Post-VRMT')

        ax12.plot(np.arange(length_C), interp1d(np.linspace(0, 1, len(data_reference_left_vel)), data_reference_left_vel)(np.linspace(0, 1, length_C)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        ax12.set_ylabel('Linear Velocity', fontsize=65, labelpad=10)
        ax12.set_xlabel('Samples', fontsize=65, labelpad=22)
        ax12.text(0.5, 1.085,  f'Left {body_velocity}', fontsize=65, ha='center', va='center', transform=ax12.transAxes)
        ax12.grid(True)
        ax12.tick_params(axis='x', labelsize=65)
        ax12.tick_params(axis='y', labelsize=65)
        h, l = ax12.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if activity == 1:
            ax12.set_ylim([min_value-0.35, max_value+0.35])
        else:
            ax12.set_ylim([min_value-0.18, max_value+0.19])

        if activity == 1:
            left = 0.065 if not all_participants else 0.06
            wspace = 0.22
        else:
            left = 0.065 if not all_participants else 0.075
            wspace = 0.22 if not all_participants else 0.28
        right = 0.98 if not all_participants else 0.99
        hspace = 0.4 if not all_participants else 0.45
        top = 0.91 if not all_participants else 0.94
        bottom = 0.11 if not all_participants else 0.09
    
    if activity == 2:
        max_value = max(np.max(mean_curves_A_PRE[1]), np.max(mean_curves_A_POS[1]), np.max(mean_curves_B_PRE[1]), np.max(mean_curves_B_POS[1]), np.max(mean_curves_C_PRE[1]), np.max(mean_curves_C_POS[1]))
        min_value = min(np.min(mean_curves_A_PRE[1]), np.min(mean_curves_A_POS[1]), np.min(mean_curves_B_PRE[1]), np.min(mean_curves_B_POS[1]), np.min(mean_curves_C_PRE[1]), np.min(mean_curves_C_POS[1]))

        ax1.plot(mean_curves_A_PRE[0], mean_curves_A_PRE[1], '-', label=f'Mean Pre-ME', color=plt.cm.Reds(0.7), linewidth=3.0)
        ax1.plot(mean_curves_A_POS[0], mean_curves_A_POS[1], '-', label=f'Mean Pos-ME', color=plt.cm.Greens(0.85), linewidth=3.0)
        ax1.plot(interp1d(np.linspace(0, 1, len(ACT2_COM_X)), ACT2_COM_X)(np.linspace(0, 1, 331)), interp1d(np.linspace(0, 1, len(ACT2_COM_Y)), ACT2_COM_Y)(np.linspace(0, 1, 331)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)

        ax1.set_xlabel('Mediolateral Position', fontsize=35, labelpad=22)
        ax1.set_ylabel('Anteroposterior Position', fontsize=35, labelpad=7)
        ax1.text(0.5, 1.11,  f'Session A', fontsize=35, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.04, 'Center of Mass', fontsize=35, ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=35, pad=25)
        ax1.tick_params(axis='y', labelsize=35)
        ax1.set_ylim([min_value-0.01, max_value+0.01])

        ax2.plot(mean_curves_B_PRE[0], mean_curves_B_PRE[1], '-', label=f'Mean Pre-ME', color=plt.cm.Reds(0.7), linewidth=3.0)
        ax2.plot(mean_curves_B_POS[0], mean_curves_B_POS[1], '-', label=f'Mean Pos-ME', color=plt.cm.Greens(0.85), linewidth=3.0)
        ax2.plot(interp1d(np.linspace(0, 1, len(ACT2_COM_X)), ACT2_COM_X)(np.linspace(0, 1, 331)), interp1d(np.linspace(0, 1, len(ACT2_COM_Y)), ACT2_COM_Y)(np.linspace(0, 1, 331)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)
        ax2.set_xlabel('Mediolateral Position', fontsize=35, labelpad=22)
        ax2.set_ylabel('Anteroposterior Position', fontsize=35, labelpad=7)
        ax2.text(0.5, 1.11,  f'Session B', fontsize=35, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 1.04, 'Center of Mass', fontsize=35, ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', labelsize=35, pad=25)
        ax2.tick_params(axis='y', labelsize=35)
        ax2.set_ylim([min_value-0.01, max_value+0.01])

        ax3.plot(mean_curves_C_PRE[0], mean_curves_C_PRE[1], '-', label=f'Mean Pre-ME', color=plt.cm.Reds(0.7), linewidth=3.0)
        #ax3.plot(mean_curves_C_DURING[0], mean_curves_C_DURING[1], '-', label=f'Mean During-ME', color=plt.cm.Blues(1.0))
        ax3.plot(mean_curves_C_POS[0], mean_curves_C_POS[1], '-', label=f'Mean Pos-ME', color=plt.cm.Greens(0.85), linewidth=3.0)
        ax3.plot(interp1d(np.linspace(0, 1, len(ACT2_COM_X)), ACT2_COM_X)(np.linspace(0, 1, 331)), interp1d(np.linspace(0, 1, len(ACT2_COM_Y)), ACT2_COM_Y)(np.linspace(0, 1, 331)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)
        ax3.set_xlabel('Mediolateral Position', fontsize=35, labelpad=22)
        ax3.set_ylabel('Anteroposterior Position', fontsize=35, labelpad=7)
        ax3.text(0.5, 1.11,  f'Session C', fontsize=35, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
        ax3.text(0.5, 1.04, 'Center of Mass', fontsize=35, ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True)
        ax3.tick_params(axis='x', labelsize=35, pad=25)
        ax3.tick_params(axis='y', labelsize=35)
        ax3.set_ylim([min_value-0.01, max_value+0.01])

        left = 0.05 if not all_participants else 0.07
        right = 0.97 if not all_participants else 0.99
        hspace = 0.3 if not all_participants else 0.3
        wspace = 0.2 if not all_participants else 0.27
        top = 0.88 if not all_participants else 0.895
        bottom = 0.17 if not all_participants else 0.22
    
    if activity == 31:
        max_value = max(np.max(mean_curves_A_PRE[0]), np.max(mean_curves_A_POS[0]), np.max(mean_curves_A_PRE[3]), np.max(mean_curves_A_POS[3]), np.max(mean_curves_B_PRE[0]), np.max(mean_curves_B_POS[0]), np.max(mean_curves_B_PRE[3]), np.max(mean_curves_B_POS[3]), np.max(mean_curves_C_PRE[0]), np.max(mean_curves_C_POS[0]), np.max(mean_curves_C_PRE[3]), np.max(mean_curves_C_POS[3]))
        min_value = min(np.min(mean_curves_A_PRE[0]), np.min(mean_curves_A_POS[0]), np.min(mean_curves_A_PRE[3]), np.min(mean_curves_A_POS[3]), np.min(mean_curves_B_PRE[0]), np.min(mean_curves_B_POS[0]), np.min(mean_curves_B_PRE[3]), np.min(mean_curves_B_POS[3]), np.min(mean_curves_C_PRE[0]), np.min(mean_curves_C_POS[0]), np.min(mean_curves_C_PRE[3]), np.min(mean_curves_C_POS[3]))

        ax1.plot(np.arange(len(mean_curves_A_PRE[0])), mean_curves_A_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax1.plot(np.arange(len(mean_curves_A_POS[0])), mean_curves_A_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax1.plot(np.arange(len(mean_curves_A_POS[0])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_X)), ACT31_RIGHT_HAND_X)(np.linspace(0, 1, len(mean_curves_A_POS[0]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower0 = np.array(mean_curves_A_PRE[0]) - np.array(std_dev_curves_A_PRE[0])
            upper0 = np.array(mean_curves_A_PRE[0]) + np.array(std_dev_curves_A_PRE[0])
            ax1.fill_between(np.arange(len(mean_curves_A_PRE[0])), lower0, upper0, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower1 = np.array(mean_curves_A_POS[0]) - np.array(std_dev_curves_A_POS[0])
            upper1 = np.array(mean_curves_A_POS[0]) + np.array(std_dev_curves_A_POS[0])
            ax1.fill_between(np.arange(len(mean_curves_A_POS[0])), lower1, upper1, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax1.set_ylabel('Position', fontsize=90, labelpad=22)
        ax1.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax1.text(0.5, 1.1215,  'Right Hand X', fontsize=90, ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.3, 'Session A', fontsize=90, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=80)
        ax1.tick_params(axis='y', labelsize=80)
        ax1.set_ylim([min_value-0.0015, max_value+0.0015])

        ax2.plot(np.arange(len(mean_curves_B_PRE[0])), mean_curves_B_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax2.plot(np.arange(len(mean_curves_B_POS[0])), mean_curves_B_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax2.plot(np.arange(len(mean_curves_B_POS[0])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_X)), ACT31_RIGHT_HAND_X)(np.linspace(0, 1, len(mean_curves_B_POS[0]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower4 = np.array(mean_curves_B_PRE[0]) - np.array(std_dev_curves_B_PRE[0])
            upper4 = np.array(mean_curves_B_PRE[0]) + np.array(std_dev_curves_B_PRE[0])
            ax2.fill_between(np.arange(len(mean_curves_B_PRE[0])), lower4, upper4, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower5 = np.array(mean_curves_B_POS[0]) - np.array(std_dev_curves_B_POS[0])
            upper5 = np.array(mean_curves_B_POS[0]) + np.array(std_dev_curves_B_POS[0])
            ax2.fill_between(np.arange(len(mean_curves_B_POS[0])), lower5, upper5, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax2.set_ylabel('Position', fontsize=90, labelpad=22)
        ax2.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax2.text(0.5, 1.1215,  'Right Hand X', fontsize=90, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 1.3, 'Session B', fontsize=90, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', labelsize=80)
        ax2.tick_params(axis='y', labelsize=80)
        ax2.set_ylim([min_value-0.0015, max_value+0.0015])

        ax3.plot(np.arange(len(mean_curves_C_PRE[0])), mean_curves_C_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        #ax3.plot(np.arange(len(mean_curves_C_DURING[0])), mean_curves_C_DURING[0], '-', label=f'Right During-ME', color=plt.cm.Purples(1.0), linewidth=2.5)
        ax3.plot(np.arange(len(mean_curves_C_POS[0])), mean_curves_C_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax3.plot(np.arange(len(mean_curves_C_POS[0])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_X)), ACT31_RIGHT_HAND_X)(np.linspace(0, 1, len(mean_curves_C_POS[0]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower8 = np.array(mean_curves_C_PRE[0]) - np.array(std_dev_curves_C_PRE[0])
            upper8 = np.array(mean_curves_C_PRE[0]) + np.array(std_dev_curves_C_PRE[0])
            ax3.fill_between(np.arange(len(mean_curves_C_PRE[0])), lower8, upper8, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            #lower9 = np.array(mean_curves_C_DURING[0]) - np.array(std_dev_curves_C_DURING[0])
            #upper9 = np.array(mean_curves_C_DURING[0]) + np.array(std_dev_curves_C_DURING[0])
            #ax3.fill_between(np.arange(len(mean_curves_C_DURING[0])), lower9, upper9, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. Right During-ME')
            lower10 = np.array(mean_curves_C_POS[0]) - np.array(std_dev_curves_C_POS[0])
            upper10 = np.array(mean_curves_C_POS[0]) + np.array(std_dev_curves_C_POS[0])
            ax3.fill_between(np.arange(len(mean_curves_C_POS[0])), lower10, upper10, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax3.set_ylabel('Position', fontsize=90, labelpad=22)
        ax3.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax3.text(0.5, 1.1215,  'Right Hand X', fontsize=90, ha='center', va='center', transform=ax3.transAxes)
        ax3.text(0.5, 1.3, 'Session C', fontsize=90, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True)
        ax3.tick_params(axis='x', labelsize=80)
        ax3.tick_params(axis='y', labelsize=80)
        #ax3.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=1, fontsize=35)
        #ax3.set_ylim([min_value-0.06, max_value+0.08])
        ax3.set_ylim([min_value-0.0015, max_value+0.0015])


        ax4.plot(np.arange(len(mean_curves_A_PRE[3])), mean_curves_A_PRE[3], '-', label=f'Left Pre-ME', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax4.plot(np.arange(len(mean_curves_A_POS[3])), mean_curves_A_POS[3], '-', label=f'Left Pos-ME', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax4.plot(np.arange(len(mean_curves_A_POS[3])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_X)), ACT31_LEFT_HAND_X)(np.linspace(0, 1, len(mean_curves_A_POS[3]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower2 = np.array(mean_curves_A_PRE[3]) - np.array(std_dev_curves_A_PRE[3])
            upper2 = np.array(mean_curves_A_PRE[3]) + np.array(std_dev_curves_A_PRE[3])
            ax4.fill_between(np.arange(len(mean_curves_A_PRE[3])), lower2, upper2, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower3 = np.array(mean_curves_A_POS[3]) - np.array(std_dev_curves_A_POS[3])
            upper3 = np.array(mean_curves_A_POS[3]) + np.array(std_dev_curves_A_POS[3])
            ax4.fill_between(np.arange(len(mean_curves_A_POS[3])), lower3, upper3, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')
        
        ax4.set_ylabel('Position', fontsize=90, labelpad=22)
        ax4.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax4.text(0.5, 1.1215,  'Left Hand X', fontsize=90, ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True)
        ax4.tick_params(axis='x', labelsize=80)
        ax4.tick_params(axis='y', labelsize=80)
        ax4.set_ylim([min_value-0.0015, max_value+0.0015])

        ax5.plot(np.arange(len(mean_curves_B_PRE[3])), mean_curves_B_PRE[3], '-', label=f'Left Pre-ME', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax5.plot(np.arange(len(mean_curves_B_POS[3])), mean_curves_B_POS[3], '-', label=f'Left Pos-ME', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax5.plot(np.arange(len(mean_curves_B_POS[3])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_X)), ACT31_LEFT_HAND_X)(np.linspace(0, 1, len(mean_curves_B_POS[3]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower6 = np.array(mean_curves_B_PRE[3]) - np.array(std_dev_curves_B_PRE[3])
            upper6 = np.array(mean_curves_B_PRE[3]) + np.array(std_dev_curves_B_PRE[3])
            ax5.fill_between(np.arange(len(mean_curves_B_PRE[3])), lower6, upper6, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower7 = np.array(mean_curves_B_POS[3]) - np.array(std_dev_curves_B_POS[3])
            upper7 = np.array(mean_curves_B_POS[3]) + np.array(std_dev_curves_B_POS[3])
            ax5.fill_between(np.arange(len(mean_curves_B_POS[3])), lower7, upper7, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax5.set_ylabel('Position', fontsize=90, labelpad=22)
        ax5.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax5.text(0.5, 1.1215,  'Left Hand X', fontsize=90, ha='center', va='center', transform=ax5.transAxes)
        ax5.grid(True)
        ax5.tick_params(axis='x', labelsize=80)
        ax5.tick_params(axis='y', labelsize=80)
        ax5.set_ylim([min_value-0.0015, max_value+0.0015])

        ax6.plot(np.arange(len(mean_curves_C_PRE[3])), mean_curves_C_PRE[3], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax6.plot(np.arange(len(mean_curves_C_POS[3])), mean_curves_C_POS[3], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax6.plot(np.arange(len(mean_curves_C_POS[3])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_X)), ACT31_LEFT_HAND_X)(np.linspace(0, 1, len(mean_curves_C_POS[3]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower11 = np.array(mean_curves_C_PRE[3]) - np.array(std_dev_curves_C_PRE[3])
            upper11 = np.array(mean_curves_C_PRE[3]) + np.array(std_dev_curves_C_PRE[3])
            ax6.fill_between(np.arange(len(mean_curves_C_PRE[3])), lower11, upper11, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower13 = np.array(mean_curves_C_POS[3]) - np.array(std_dev_curves_C_POS[3])
            upper13 = np.array(mean_curves_C_POS[3]) + np.array(std_dev_curves_C_POS[3])
            ax6.fill_between(np.arange(len(mean_curves_C_POS[3])), lower13, upper13, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax6.set_ylabel('Position', fontsize=90, labelpad=22)
        ax6.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax6.text(0.5, 1.1215,  'Left Hand X', fontsize=90, ha='center', va='center', transform=ax6.transAxes)
        ax6.tick_params(axis='x', labelsize=80)
        ax6.tick_params(axis='y', labelsize=80)
        ax6.set_ylim([min_value-0.0015, max_value+0.0015])

        # -------------------------------------- Position Y -----------------------------------------
        max_value = max(np.max(mean_curves_A_PRE[1]), np.max(mean_curves_A_POS[1]), np.max(mean_curves_A_PRE[4]), np.max(mean_curves_A_POS[4]), np.max(mean_curves_B_PRE[1]), np.max(mean_curves_B_POS[1]), np.max(mean_curves_B_PRE[4]), np.max(mean_curves_B_POS[4]), np.max(mean_curves_C_PRE[1]), np.max(mean_curves_C_POS[1]), np.max(mean_curves_C_PRE[4]), np.max(mean_curves_C_POS[4]))
        min_value = min(np.min(mean_curves_A_PRE[1]), np.min(mean_curves_A_POS[1]), np.min(mean_curves_A_PRE[4]), np.min(mean_curves_A_POS[4]), np.min(mean_curves_B_PRE[1]), np.min(mean_curves_B_POS[1]), np.min(mean_curves_B_PRE[4]), np.min(mean_curves_B_POS[4]), np.min(mean_curves_C_PRE[1]), np.min(mean_curves_C_POS[1]), np.min(mean_curves_C_PRE[4]), np.min(mean_curves_C_POS[4]))

        ax7.plot(np.arange(len(mean_curves_A_PRE[1])), mean_curves_A_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax7.plot(np.arange(len(mean_curves_A_POS[1])), mean_curves_A_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax7.plot(np.arange(len(mean_curves_A_POS[1])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Y)), ACT31_RIGHT_HAND_Y)(np.linspace(0, 1, len(mean_curves_A_POS[1]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower1256 = np.array(mean_curves_A_PRE[1]) - np.array(std_dev_curves_A_PRE[1])
            upper1256 = np.array(mean_curves_A_PRE[1]) + np.array(std_dev_curves_A_PRE[1])
            ax7.fill_between(np.arange(len(mean_curves_A_PRE[1])), lower1256, upper1256, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower6754 = np.array(mean_curves_A_POS[1]) - np.array(std_dev_curves_A_POS[1])
            upper6754 = np.array(mean_curves_A_POS[1]) + np.array(std_dev_curves_A_POS[1])
            ax7.fill_between(np.arange(len(mean_curves_A_POS[1])), lower6754, upper6754, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax7.set_ylabel('Position', fontsize=90, labelpad=22)
        ax7.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax7.text(0.5, 1.1215,  'Right Hand Y', fontsize=90, ha='center', va='center', transform=ax7.transAxes)
        ax7.grid(True)
        ax7.tick_params(axis='x', labelsize=80)
        ax7.tick_params(axis='y', labelsize=80)
        ax7.set_ylim([min_value-0.0015, max_value+0.0015])

        ax8.plot(np.arange(len(mean_curves_B_PRE[1])), mean_curves_B_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax8.plot(np.arange(len(mean_curves_B_POS[1])), mean_curves_B_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax8.plot(np.arange(len(mean_curves_B_POS[1])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Y)), ACT31_RIGHT_HAND_Y)(np.linspace(0, 1, len(mean_curves_B_POS[1]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower41 = np.array(mean_curves_B_PRE[1]) - np.array(std_dev_curves_B_PRE[1])
            upper41 = np.array(mean_curves_B_PRE[1]) + np.array(std_dev_curves_B_PRE[1])
            ax8.fill_between(np.arange(len(mean_curves_B_PRE[1])), lower41, upper41, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower51 = np.array(mean_curves_B_POS[1]) - np.array(std_dev_curves_B_POS[1])
            upper51 = np.array(mean_curves_B_POS[1]) + np.array(std_dev_curves_B_POS[1])
            ax8.fill_between(np.arange(len(mean_curves_B_POS[1])), lower51, upper51, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax8.set_ylabel('Position', fontsize=90, labelpad=22)
        ax8.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax8.text(0.5, 1.1215,  'Right Hand Y', fontsize=90, ha='center', va='center', transform=ax8.transAxes)
        ax8.grid(True)
        ax8.tick_params(axis='x', labelsize=80)
        ax8.tick_params(axis='y', labelsize=80)
        ax8.set_ylim([min_value-0.0015, max_value+0.0015])

        ax9.plot(np.arange(len(mean_curves_C_PRE[1])), mean_curves_C_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        #ax6.plot(np.arange(len(mean_curves_C_DURING[1])), mean_curves_C_DURING[1], '-', label=f'Right During-ME', color=plt.cm.Purples(1.0), linewidth=2.5)
        ax9.plot(np.arange(len(mean_curves_C_POS[1])), mean_curves_C_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax9.plot(np.arange(len(mean_curves_C_POS[1])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Y)), ACT31_RIGHT_HAND_Y)(np.linspace(0, 1, len(mean_curves_C_POS[1]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower81 = np.array(mean_curves_C_PRE[1]) - np.array(std_dev_curves_C_PRE[1])
            upper81 = np.array(mean_curves_C_PRE[1]) + np.array(std_dev_curves_C_PRE[1])
            ax9.fill_between(np.arange(len(mean_curves_C_PRE[1])), lower81, upper81, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            #lower91 = np.array(mean_curves_C_DURING[1]) - np.array(std_dev_curves_C_DURING[1])
            #upper91 = np.array(mean_curves_C_DURING[1]) + np.array(std_dev_curves_C_DURING[1])
            #ax6.fill_between(np.arange(len(mean_curves_C_DURING[1])), lower91, upper91, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. Right During-ME')
            lower101 = np.array(mean_curves_C_POS[1]) - np.array(std_dev_curves_C_POS[1])
            upper101 = np.array(mean_curves_C_POS[1]) + np.array(std_dev_curves_C_POS[1])
            ax9.fill_between(np.arange(len(mean_curves_C_POS[1])), lower101, upper101, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax9.set_ylabel('Position', fontsize=90, labelpad=22)
        ax9.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax9.text(0.5, 1.1215,  'Right Hand Y', fontsize=90, ha='center', va='center', transform=ax9.transAxes)
        ax9.grid(True)
        ax9.tick_params(axis='x', labelsize=80)
        ax9.tick_params(axis='y', labelsize=80)
        ax9.set_ylim([min_value-0.0015, max_value+0.0015])


        ax10.plot(np.arange(len(mean_curves_A_PRE[4])), mean_curves_A_PRE[4], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax10.plot(np.arange(len(mean_curves_A_POS[4])), mean_curves_A_POS[4], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax10.plot(np.arange(len(mean_curves_A_POS[4])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Y)), ACT31_LEFT_HAND_Y)(np.linspace(0, 1, len(mean_curves_A_POS[4]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower21 = np.array(mean_curves_A_PRE[4]) - np.array(std_dev_curves_A_PRE[4])
            upper21 = np.array(mean_curves_A_PRE[4]) + np.array(std_dev_curves_A_PRE[4])
            ax10.fill_between(np.arange(len(mean_curves_A_PRE[4])), lower21, upper21, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower31 = np.array(mean_curves_A_POS[4]) - np.array(std_dev_curves_A_POS[4])
            upper31 = np.array(mean_curves_A_POS[4]) + np.array(std_dev_curves_A_POS[4])
            ax10.fill_between(np.arange(len(mean_curves_A_POS[4])), lower31, upper31, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax10.set_ylabel('Position', fontsize=90, labelpad=22)
        ax10.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax10.text(0.5, 1.1215,  'Left Hand Y', fontsize=90, ha='center', va='center', transform=ax10.transAxes)
        ax10.grid(True)
        ax10.tick_params(axis='x', labelsize=80)
        ax10.tick_params(axis='y', labelsize=80)
        ax10.set_ylim([min_value-0.0015, max_value+0.0015])

        ax11.plot(np.arange(len(mean_curves_B_PRE[4])), mean_curves_B_PRE[4], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax11.plot(np.arange(len(mean_curves_B_POS[4])), mean_curves_B_POS[4], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax11.plot(np.arange(len(mean_curves_B_POS[4])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Y)), ACT31_LEFT_HAND_Y)(np.linspace(0, 1, len(mean_curves_B_POS[4]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower61 = np.array(mean_curves_B_PRE[4]) - np.array(std_dev_curves_B_PRE[4])
            upper61 = np.array(mean_curves_B_PRE[4]) + np.array(std_dev_curves_B_PRE[4])
            ax11.fill_between(np.arange(len(mean_curves_B_PRE[4])), lower61, upper61, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower71 = np.array(mean_curves_B_POS[4]) - np.array(std_dev_curves_B_POS[4])
            upper71 = np.array(mean_curves_B_POS[4]) + np.array(std_dev_curves_B_POS[4])
            ax11.fill_between(np.arange(len(mean_curves_B_POS[4])), lower71, upper71, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')
        
        ax11.set_ylabel('Position', fontsize=90, labelpad=22)
        ax11.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax11.text(0.5, 1.1215,  'Left Hand Y', fontsize=90, ha='center', va='center', transform=ax11.transAxes)
        ax11.grid(True)
        ax11.tick_params(axis='x', labelsize=80)
        ax11.tick_params(axis='y', labelsize=80)
        ax11.set_ylim([min_value-0.0015, max_value+0.0015])

        ax12.plot(np.arange(len(mean_curves_C_PRE[4])), mean_curves_C_PRE[4], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax12.plot(np.arange(len(mean_curves_C_POS[4])), mean_curves_C_POS[4], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax12.plot(np.arange(len(mean_curves_C_POS[4])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Y)), ACT31_LEFT_HAND_Y)(np.linspace(0, 1, len(mean_curves_C_POS[4]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower111 = np.array(mean_curves_C_PRE[4]) - np.array(std_dev_curves_C_PRE[4])
            upper111 = np.array(mean_curves_C_PRE[4]) + np.array(std_dev_curves_C_PRE[4])
            ax12.fill_between(np.arange(len(mean_curves_C_PRE[4])), lower111, upper111, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower131 = np.array(mean_curves_C_POS[4]) - np.array(std_dev_curves_C_POS[4])
            upper131 = np.array(mean_curves_C_POS[4]) + np.array(std_dev_curves_C_POS[4])
            ax12.fill_between(np.arange(len(mean_curves_C_POS[4])), lower131, upper131, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax12.set_ylabel('Position', fontsize=90, labelpad=22)
        ax12.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax12.text(0.5, 1.1215,  'Left Hand Y', fontsize=90, ha='center', va='center', transform=ax12.transAxes)
        ax12.grid(True)
        ax12.tick_params(axis='x', labelsize=80)
        ax12.tick_params(axis='y', labelsize=80)
        ax12.set_ylim([min_value-0.0015, max_value+0.0015])

        # -------------------------------------- Position Z -----------------------------------------
        max_value = max(np.max(mean_curves_A_PRE[2]), np.max(mean_curves_A_POS[2]), np.max(mean_curves_A_PRE[5]), np.max(mean_curves_A_POS[5]), np.max(mean_curves_B_PRE[2]), np.max(mean_curves_B_POS[2]), np.max(mean_curves_B_PRE[5]), np.max(mean_curves_B_POS[5]), np.max(mean_curves_C_PRE[2]), np.max(mean_curves_C_POS[2]), np.max(mean_curves_C_PRE[5]), np.max(mean_curves_C_POS[5]))
        min_value = min(np.min(mean_curves_A_PRE[2]), np.min(mean_curves_A_POS[2]), np.min(mean_curves_A_PRE[5]), np.min(mean_curves_A_POS[5]), np.min(mean_curves_B_PRE[2]), np.min(mean_curves_B_POS[2]), np.min(mean_curves_B_PRE[5]), np.min(mean_curves_B_POS[5]), np.min(mean_curves_C_PRE[2]), np.min(mean_curves_C_POS[2]), np.min(mean_curves_C_PRE[5]), np.min(mean_curves_C_POS[5]))

        ax13.plot(np.arange(len(mean_curves_A_PRE[2])), mean_curves_A_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax13.plot(np.arange(len(mean_curves_A_POS[2])), mean_curves_A_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax13.plot(np.arange(len(mean_curves_A_POS[2])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Z)), ACT31_RIGHT_HAND_Z)(np.linspace(0, 1, len(mean_curves_A_POS[2]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower011 = np.array(mean_curves_A_PRE[2]) - np.array(std_dev_curves_A_PRE[2])
            upper011 = np.array(mean_curves_A_PRE[2]) + np.array(std_dev_curves_A_PRE[2])
            ax13.fill_between(np.arange(len(mean_curves_A_PRE[2])), lower011, upper011, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower021 = np.array(mean_curves_A_POS[2]) - np.array(std_dev_curves_A_POS[2])
            upper021 = np.array(mean_curves_A_POS[2]) + np.array(std_dev_curves_A_POS[2])
            ax13.fill_between(np.arange(len(mean_curves_A_POS[2])), lower021, upper021, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax13.set_ylabel('Position', fontsize=90, labelpad=22)
        ax13.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax13.text(0.5, 1.1215,  'Right Hand Z', fontsize=90, ha='center', va='center', transform=ax13.transAxes)
        ax13.grid(True)
        ax13.tick_params(axis='x', labelsize=80)
        ax13.tick_params(axis='y', labelsize=80)
        ax13.set_ylim([min_value-0.0015, max_value+0.0015])

        ax14.plot(np.arange(len(mean_curves_B_PRE[2])), mean_curves_B_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        ax14.plot(np.arange(len(mean_curves_B_POS[2])), mean_curves_B_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax14.plot(np.arange(len(mean_curves_B_POS[2])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Z)), ACT31_RIGHT_HAND_Z)(np.linspace(0, 1, len(mean_curves_B_POS[2]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower42 = np.array(mean_curves_B_PRE[2]) - np.array(std_dev_curves_B_PRE[2])
            upper42 = np.array(mean_curves_B_PRE[2]) + np.array(std_dev_curves_B_PRE[2])
            ax14.fill_between(np.arange(len(mean_curves_B_PRE[2])), lower42, upper42, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower52 = np.array(mean_curves_B_POS[2]) - np.array(std_dev_curves_B_POS[2])
            upper52 = np.array(mean_curves_B_POS[2]) + np.array(std_dev_curves_B_POS[2])
            ax14.fill_between(np.arange(len(mean_curves_B_POS[2])), lower52, upper52, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax14.set_ylabel('Position', fontsize=90, labelpad=22)
        ax14.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax14.text(0.5, 1.1215,  'Right Hand Z', fontsize=90, ha='center', va='center', transform=ax14.transAxes)
        ax14.grid(True)
        ax14.tick_params(axis='x', labelsize=80)
        ax14.tick_params(axis='y', labelsize=80)
        ax14.set_ylim([min_value-0.0015, max_value+0.0015])

        ax15.plot(np.arange(len(mean_curves_C_PRE[2])), mean_curves_C_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=6.0)
        #ax9.plot(np.arange(len(mean_curves_C_DURING[2])), mean_curves_C_DURING[2], '-', label=f'Right During-ME', color=plt.cm.Purples(1.0), linewidth=2.5)
        ax15.plot(np.arange(len(mean_curves_C_POS[2])), mean_curves_C_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=6.0)
        ax15.plot(np.arange(len(mean_curves_C_POS[2])), interp1d(np.linspace(0, 1, len(ACT31_RIGHT_HAND_Z)), ACT31_RIGHT_HAND_Z)(np.linspace(0, 1, len(mean_curves_C_POS[2]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower82 = np.array(mean_curves_C_PRE[2]) - np.array(std_dev_curves_C_PRE[2])
            upper82 = np.array(mean_curves_C_PRE[2]) + np.array(std_dev_curves_C_PRE[2])
            ax15.fill_between(np.arange(len(mean_curves_C_PRE[2])), lower82, upper82, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            #lower92 = np.array(mean_curves_C_DURING[2]) - np.array(std_dev_curves_C_DURING[2])
            #upper92 = np.array(mean_curves_C_DURING[2]) + np.array(std_dev_curves_C_DURING[2])
            #ax9.fill_between(np.arange(len(mean_curves_C_DURING[2])), lower92, upper92, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. Right During-ME')
            lower102 = np.array(mean_curves_C_POS[2]) - np.array(std_dev_curves_C_POS[2])
            upper102 = np.array(mean_curves_C_POS[2]) + np.array(std_dev_curves_C_POS[2])
            ax15.fill_between(np.arange(len(mean_curves_C_POS[2])), lower102, upper102, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax15.set_ylabel('Position', fontsize=90, labelpad=22)
        ax15.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax15.text(0.5, 1.1215,  'Right Hand Z', fontsize=90, ha='center', va='center', transform=ax15.transAxes)
        ax15.grid(True)
        ax15.tick_params(axis='x', labelsize=80)
        ax15.tick_params(axis='y', labelsize=80)
        ax15.set_ylim([min_value-0.0015, max_value+0.0015])


        ax16.plot(np.arange(len(mean_curves_A_PRE[5])), mean_curves_A_PRE[5], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax16.plot(np.arange(len(mean_curves_A_POS[5])), mean_curves_A_POS[5], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax16.plot(np.arange(len(mean_curves_A_POS[5])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Z)), ACT31_LEFT_HAND_Z)(np.linspace(0, 1, len(mean_curves_A_POS[5]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower22 = np.array(mean_curves_A_PRE[5]) - np.array(std_dev_curves_A_PRE[5])
            upper22 = np.array(mean_curves_A_PRE[5]) + np.array(std_dev_curves_A_PRE[5])
            ax16.fill_between(np.arange(len(mean_curves_A_PRE[5])), lower22, upper22, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower32 = np.array(mean_curves_A_POS[5]) - np.array(std_dev_curves_A_POS[5])
            upper32 = np.array(mean_curves_A_POS[5]) + np.array(std_dev_curves_A_POS[5])
            ax16.fill_between(np.arange(len(mean_curves_A_POS[5])), lower32, upper32, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax16.set_ylabel('Position', fontsize=90, labelpad=22)
        ax16.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax16.text(0.5, 1.1215,  'Left Hand Z', fontsize=90, ha='center', va='center', transform=ax16.transAxes)
        ax16.grid(True)
        ax16.tick_params(axis='x', labelsize=80)
        ax16.tick_params(axis='y', labelsize=80)
        ax16.set_ylim([min_value-0.0015, max_value+0.0015])

        ax17.plot(np.arange(len(mean_curves_B_PRE[5])), mean_curves_B_PRE[5], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax17.plot(np.arange(len(mean_curves_B_POS[5])), mean_curves_B_POS[5], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax17.plot(np.arange(len(mean_curves_B_POS[5])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Z)), ACT31_LEFT_HAND_Z)(np.linspace(0, 1, len(mean_curves_B_POS[5]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower62 = np.array(mean_curves_B_PRE[5]) - np.array(std_dev_curves_B_PRE[5])
            upper62 = np.array(mean_curves_B_PRE[5]) + np.array(std_dev_curves_B_PRE[5])
            ax17.fill_between(np.arange(len(mean_curves_B_PRE[5])), lower62, upper62, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower72 = np.array(mean_curves_B_POS[5]) - np.array(std_dev_curves_B_POS[5])
            upper72 = np.array(mean_curves_B_POS[5]) + np.array(std_dev_curves_B_POS[5])
            ax17.fill_between(np.arange(len(mean_curves_B_POS[5])), lower72, upper72, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax17.set_ylabel('Position', fontsize=90, labelpad=22)
        ax17.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax17.text(0.5, 1.1215,  'Left Hand Z', fontsize=90, ha='center', va='center', transform=ax17.transAxes)
        ax17.grid(True)
        ax17.tick_params(axis='x', labelsize=80)
        ax17.tick_params(axis='y', labelsize=80)
        ax17.set_ylim([min_value-0.0015, max_value+0.0015])

        ax18.plot(np.arange(len(mean_curves_C_PRE[5])), mean_curves_C_PRE[5], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=7.0)
        ax18.plot(np.arange(len(mean_curves_C_POS[5])), mean_curves_C_POS[5], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=7.0)
        ax18.plot(np.arange(len(mean_curves_C_POS[5])), interp1d(np.linspace(0, 1, len(ACT31_LEFT_HAND_Z)), ACT31_LEFT_HAND_Z)(np.linspace(0, 1, len(mean_curves_C_POS[5]))), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=6.0)

        if all_participants:
            lower112 = np.array(mean_curves_C_PRE[5]) - np.array(std_dev_curves_C_PRE[5])
            upper112 = np.array(mean_curves_C_PRE[5]) + np.array(std_dev_curves_C_PRE[5])
            ax18.fill_between(np.arange(len(mean_curves_C_PRE[5])), lower112, upper112, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
            lower132 = np.array(mean_curves_C_POS[5]) - np.array(std_dev_curves_C_POS[5])
            upper132 = np.array(mean_curves_C_POS[5]) + np.array(std_dev_curves_C_POS[5])
            ax18.fill_between(np.arange(len(mean_curves_C_POS[5])), lower132, upper132, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')

        ax18.set_ylabel('Position', fontsize=90, labelpad=22)
        ax18.set_xlabel('Samples', fontsize=90, labelpad=22)
        ax18.text(0.5, 1.1215,  'Left Hand Z', fontsize=90, ha='center', va='center', transform=ax18.transAxes)
        ax18.grid(True)
        ax18.tick_params(axis='x', labelsize=80)
        ax18.tick_params(axis='y', labelsize=80)
        h, l = ax18.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax18.set_ylim([min_value-0.0015, max_value+0.0015])

        left = 0.085 if not all_participants else 0.095
        right = 0.95 if not all_participants else 0.96
        hspace = 0.65 if not all_participants else 0.65
        wspace = 0.38 if not all_participants else 0.39
        top = 0.915 if not all_participants else 0.96
        bottom = 0.11 if not all_participants else 0.07

    folder = "Healthy/final_plots" if all_participants else f"Healthy/participant_{participant+1}/final_plots"
    name_file = f"final_act_{activity}.png" if all_participants else f"final_participant_{participant+1}_act_{activity}.png"
    savefig = f"{folder}/final_act_{activity}.png"
    savefig_dissertation = os.path.join('..', 'Dissertation/images/final_plots', name_file)
    
    if activity == 1 or activity == 32:
        fontsize = 70
        labelsize = 65
        plt.legend(handles, labels, bbox_to_anchor=(-0.75, -0.6), loc="lower center", ncol=5, fontsize=labelsize)
    elif activity == 2:
        fontsize = 30
        labelsize = 30
        if not all_participants:
            plt.legend(bbox_to_anchor=(-0.75, -0.17), loc="lower center", ncol=3, fontsize=labelsize)
        else:
            plt.legend(bbox_to_anchor=(-0.75, -0.34), loc="lower center", ncol=3, fontsize=labelsize)
    else:
        fontsize = 80
        labelsize = 85
        if not all_participants:
            plt.legend(handles, labels, bbox_to_anchor=(-1.0, -0.95), loc="lower center", ncol=5, fontsize=labelsize)
        else:
            plt.legend(handles, labels, bbox_to_anchor=(-1.0, -0.73), loc="lower center", ncol=5, fontsize=labelsize)

    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    plt.savefig(savefig)
    plt.savefig(savefig_dissertation)
    #plt.show()

def plot_final_data_PD(mean_curves_B_PRE, std_dev_curves_B_PRE, mean_curves_B_POS, std_dev_curves_B_POS, mean_curves_B_DURING, std_dev_curves_B_DURING, participant: int, activity: int):
    if activity == 1 or activity == 32:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 15))
    elif activity == 2:
        fig, ax1 = plt.subplots(1, 1, figsize=(30, 15))
    else:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(45, 20))

    handles, labels = [], []

    body_position = ""
    body_velocity = ""
    if activity == 1:
        body_position = "Foot Z"
        body_velocity = "Foot Z"
        data_reference_right_pos = ACT1_RIGHT_FOOT_POS
        data_reference_left_pos = ACT1_LEFT_FOOT_POS 
        data_reference_right_vel = ACT1_RIGHT_FOOT_VEL
        data_reference_left_vel = ACT1_LEFT_FOOT_VEL
        length = 277
    if activity == 32:
        body_position = "Hand Z"
        body_velocity = "Hand Y"
        data_reference_right_pos = ACT32_RIGHT_HAND_POS
        data_reference_left_pos = ACT32_LEFT_HAND_POS 
        data_reference_right_vel = ACT32_RIGHT_HAND_VEL
        data_reference_left_vel = ACT32_LEFT_HAND_VEL
        length = 915

    if activity == 1 or activity == 32:
        max_value = max(np.max(mean_curves_B_PRE[0]), np.max(mean_curves_B_POS[0]), np.max(mean_curves_B_DURING[0]), np.max(mean_curves_B_PRE[1]), np.max(mean_curves_B_POS[1]), np.max(mean_curves_B_DURING[1]))
        min_value = min(np.min(mean_curves_B_PRE[0]), np.min(mean_curves_B_POS[0]), np.min(mean_curves_B_DURING[0]), np.min(mean_curves_B_PRE[1]), np.min(mean_curves_B_POS[1]), np.min(mean_curves_B_DURING[1]))

        ax1.plot(np.arange(len(mean_curves_B_PRE[0])), mean_curves_B_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax1.plot(np.arange(len(mean_curves_B_POS[0])), mean_curves_B_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        ax1.plot(np.arange(len(mean_curves_B_DURING[0])), mean_curves_B_DURING[0], '-', label=f'Mean During-VRMT', color=plt.cm.Purples(0.8), linewidth=2.0)
        if activity == 1:
            ax1.plot(np.arange(length), interp1d(np.linspace(0, 1, len(data_reference_right_pos)), data_reference_right_pos)(np.linspace(0, 1, length)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)

        lower0 = np.array(mean_curves_B_PRE[0]) - np.array(std_dev_curves_B_PRE[0])
        upper0 = np.array(mean_curves_B_PRE[0]) + np.array(std_dev_curves_B_PRE[0])
        ax1.fill_between(np.arange(len(mean_curves_B_PRE[0])), lower0, upper0, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower1 = np.array(mean_curves_B_POS[0]) - np.array(std_dev_curves_B_POS[0])
        upper1 = np.array(mean_curves_B_POS[0]) + np.array(std_dev_curves_B_POS[0])
        ax1.fill_between(np.arange(len(mean_curves_B_POS[0])), lower1, upper1, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        lower12 = np.array(mean_curves_B_DURING[0]) - np.array(std_dev_curves_B_DURING[0])
        upper12 = np.array(mean_curves_B_DURING[0]) + np.array(std_dev_curves_B_DURING[0])
        ax1.fill_between(np.arange(len(mean_curves_B_DURING[0])), lower12, upper12, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. During-VRMT')

        ax1.set_ylabel('Position', fontsize=28)
        ax1.set_xlabel('Samples', fontsize=28)
        ax1.text(0.5, 1.045,  f'Right {body_position}', fontsize=28, ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.12, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=28)
        ax1.tick_params(axis='y', labelsize=28)
        ax1.set_ylim([min_value-0.02, np.max(data_reference_right_pos)+0.005])

        ax2.plot(np.arange(len(mean_curves_B_PRE[1])), mean_curves_B_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax2.plot(np.arange(len(mean_curves_B_POS[1])), mean_curves_B_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        ax2.plot(np.arange(len(mean_curves_B_DURING[1])), mean_curves_B_DURING[1], '-', label=f'Mean During-VRMT', color=plt.cm.Purples(0.8), linewidth=2.0)
        if activity == 1:
            ax2.plot(np.arange(length), interp1d(np.linspace(0, 1, len(data_reference_left_pos)), data_reference_left_pos)(np.linspace(0, 1, length)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)

        lower2 = np.array(mean_curves_B_PRE[1]) - np.array(std_dev_curves_B_PRE[1])
        upper2 = np.array(mean_curves_B_PRE[1]) + np.array(std_dev_curves_B_PRE[1])
        ax2.fill_between(np.arange(len(mean_curves_B_PRE[1])), lower2, upper2, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower3 = np.array(mean_curves_B_POS[1]) - np.array(std_dev_curves_B_POS[1])
        upper3 = np.array(mean_curves_B_POS[1]) + np.array(std_dev_curves_B_POS[1])
        ax2.fill_between(np.arange(len(mean_curves_B_POS[1])), lower3, upper3, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        lower32 = np.array(mean_curves_B_DURING[1]) - np.array(std_dev_curves_B_DURING[1])
        upper32 = np.array(mean_curves_B_DURING[1]) + np.array(std_dev_curves_B_DURING[1])
        ax2.fill_between(np.arange(len(mean_curves_B_DURING[1])), lower32, upper32, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. During-VRMT')
        ax2.set_ylabel('Position', fontsize=28)
        ax2.set_xlabel('Samples', fontsize=28)
        ax2.text(0.5, 1.045,  f'Left {body_position}', fontsize=28, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 1.12, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', labelsize=28)
        ax2.tick_params(axis='y', labelsize=28)
        ax2.set_ylim([min_value-0.02, np.max(data_reference_left_pos)+0.005])

        max_value = max(np.max(mean_curves_B_PRE[2]), np.max(mean_curves_B_POS[2]), np.max(mean_curves_B_DURING[2]), np.max(mean_curves_B_PRE[3]), np.max(mean_curves_B_POS[3]), np.max(mean_curves_B_DURING[3]))
        min_value = min(np.min(mean_curves_B_PRE[2]), np.min(mean_curves_B_POS[2]), np.min(mean_curves_B_DURING[2]), np.min(mean_curves_B_PRE[3]), np.min(mean_curves_B_POS[3]), np.min(mean_curves_B_DURING[3]))

        ax3.plot(np.arange(len(mean_curves_B_PRE[2])), mean_curves_B_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax3.plot(np.arange(len(mean_curves_B_POS[2])), mean_curves_B_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        ax3.plot(np.arange(len(mean_curves_B_DURING[2])), mean_curves_B_DURING[2], '-', label=f'Mean During-VRMT', color=plt.cm.Purples(0.8), linewidth=2.0)
        ax3.plot(np.arange(length), interp1d(np.linspace(0, 1, len(data_reference_right_vel)), data_reference_right_vel)(np.linspace(0, 1, length)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)

        lower4 = np.array(mean_curves_B_PRE[2]) - np.array(std_dev_curves_B_PRE[2])
        upper4 = np.array(mean_curves_B_PRE[2]) + np.array(std_dev_curves_B_PRE[2])
        ax3.fill_between(np.arange(len(mean_curves_B_PRE[2])), lower4, upper4, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower5 = np.array(mean_curves_B_POS[2]) - np.array(std_dev_curves_B_POS[2])
        upper5 = np.array(mean_curves_B_POS[2]) + np.array(std_dev_curves_B_POS[2])
        ax3.fill_between(np.arange(len(mean_curves_B_POS[2])), lower5, upper5, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        lower52 = np.array(mean_curves_B_DURING[2]) - np.array(std_dev_curves_B_DURING[2])
        upper52 = np.array(mean_curves_B_DURING[2]) + np.array(std_dev_curves_B_DURING[2])
        ax3.fill_between(np.arange(len(mean_curves_B_DURING[2])), lower52, upper52, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. Post-VRMT')
        ax3.set_ylabel('Linear Velocity', fontsize=28)
        ax3.set_xlabel('Samples', fontsize=28)
        ax3.text(0.5, 1.045,  f'Right {body_velocity}', fontsize=28, ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True)
        ax3.tick_params(axis='x', labelsize=28)
        ax3.tick_params(axis='y', labelsize=28)
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if activity == 1:
            ax3.set_ylim([np.min(data_reference_right_vel)-0.01, np.max(data_reference_right_vel)+0.01])
        else:
            ax3.set_ylim([min_value-0.18, max_value+0.19])

        ax4.plot(np.arange(len(mean_curves_B_PRE[3])), mean_curves_B_PRE[3], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax4.plot(np.arange(len(mean_curves_B_POS[3])), mean_curves_B_POS[3], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        ax4.plot(np.arange(len(mean_curves_B_DURING[3])), mean_curves_B_DURING[3], '-', label=f'Mean During-VRMT', color=plt.cm.Purples(0.8), linewidth=2.0)
        ax4.plot(np.arange(length), interp1d(np.linspace(0, 1, len(data_reference_left_vel)), data_reference_left_vel)(np.linspace(0, 1, length)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)

        lower6 = np.array(mean_curves_B_PRE[3]) - np.array(std_dev_curves_B_PRE[3])
        upper6 = np.array(mean_curves_B_PRE[3]) + np.array(std_dev_curves_B_PRE[3])
        ax4.fill_between(np.arange(len(mean_curves_B_PRE[3])), lower6, upper6, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower7 = np.array(mean_curves_B_POS[3]) - np.array(std_dev_curves_B_POS[3])
        upper7 = np.array(mean_curves_B_POS[3]) + np.array(std_dev_curves_B_POS[3])
        ax4.fill_between(np.arange(len(mean_curves_B_POS[3])), lower7, upper7, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        lower72 = np.array(mean_curves_B_DURING[3]) - np.array(std_dev_curves_B_DURING[3])
        upper72 = np.array(mean_curves_B_DURING[3]) + np.array(std_dev_curves_B_DURING[3])
        ax4.fill_between(np.arange(len(mean_curves_B_DURING[3])), lower72, upper72, alpha=0.3, edgecolor=plt.cm.Purples(0.9), facecolor=plt.cm.Purples(0.6), label='Std. During-VRMT')
        ax4.set_ylabel('Linear Velocity', fontsize=28)
        ax4.set_xlabel('Samples', fontsize=28)
        ax4.text(0.5, 1.045,  f'Left {body_velocity}', fontsize=28, ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True)
        ax4.tick_params(axis='x', labelsize=28)
        ax4.tick_params(axis='y', labelsize=28)
        if activity == 1:
            ax4.set_ylim([np.min(data_reference_left_vel)-0.01, np.max(data_reference_left_vel)+0.01])
        else:
            ax4.set_ylim([min_value-0.18, max_value+0.19])

        left = 0.06
        right = 0.985
        hspace = 0.29
        wspace = 0.16
        top = 0.93
        bottom = 0.16
    
    if activity == 2:
        ax1.plot(mean_curves_B_PRE[0], mean_curves_B_PRE[1], '-', label=f'Mean Pre-ME', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax1.plot(mean_curves_B_POS[0], mean_curves_B_POS[1], '-', label=f'Mean Pos-ME', color=plt.cm.Greens(0.85), linewidth=2.0)
        ax1.plot(interp1d(np.linspace(0, 1, len(ACT2_COM_X)), ACT2_COM_X)(np.linspace(0, 1, 331)), interp1d(np.linspace(0, 1, len(ACT2_COM_Y)), ACT2_COM_Y)(np.linspace(0, 1, 331)), '--', label=f'Healthy Reference', color=plt.cm.Greys(1.0), linewidth=3.0)
        ax1.set_xlabel('Mediolateral Position', fontsize=28)
        ax1.set_ylabel('Anteroposterior Position', fontsize=28)
        lower0 = np.array(mean_curves_B_PRE[1]) - np.array(std_dev_curves_B_PRE[1])
        upper0 = np.array(mean_curves_B_PRE[1]) + np.array(std_dev_curves_B_PRE[1])
        ax1.fill_between(np.arange(len(mean_curves_B_PRE[0])), lower0, upper0, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower1 = np.array(mean_curves_B_POS[1]) - np.array(std_dev_curves_B_POS[1])
        upper1 = np.array(mean_curves_B_POS[1]) + np.array(std_dev_curves_B_POS[1])
        ax1.fill_between(np.arange(len(mean_curves_B_POS[0])), lower1, upper1, alpha=0.3, edgecolor=plt.cm.Greens(0.9), facecolor=plt.cm.Greens(0.6), label='Std. Post-VRMT')
        ax1.text(0.5, 1.085,  f'Center of Mass', fontsize=28, ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.22, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=28)
        ax1.tick_params(axis='y', labelsize=28)

        left = 0.04
        right = 0.98
        hspace = 0.3
        wspace=0.2
        top = 0.95
        bottom = 0.08

    if activity == 31:
        max_value_X = max(np.max(mean_curves_B_PRE[0]), np.max(mean_curves_B_POS[0]), np.max(mean_curves_B_PRE[3]), np.max(mean_curves_B_POS[3]))
        min_value_X = min(np.min(mean_curves_B_PRE[0]), np.min(mean_curves_B_POS[0]), np.min(mean_curves_B_PRE[3]), np.min(mean_curves_B_POS[3]))

        max_value_Y = max(np.max(mean_curves_B_PRE[1]), np.max(mean_curves_B_POS[1]), np.max(mean_curves_B_PRE[4]), np.max(mean_curves_B_POS[4]))
        min_value_Y = min(np.min(mean_curves_B_PRE[1]), np.min(mean_curves_B_POS[1]), np.min(mean_curves_B_PRE[4]), np.min(mean_curves_B_POS[4]))

        max_value_Z = max(np.max(mean_curves_B_PRE[2]), np.max(mean_curves_B_POS[2]), np.max(mean_curves_B_PRE[5]), np.max(mean_curves_B_POS[5]))
        min_value_Z = min(np.min(mean_curves_B_PRE[2]), np.min(mean_curves_B_POS[2]), np.min(mean_curves_B_PRE[5]), np.min(mean_curves_B_POS[5]))

        ax1.plot(np.arange(len(mean_curves_B_PRE[0])), mean_curves_B_PRE[0], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax1.plot(np.arange(len(mean_curves_B_POS[0])), mean_curves_B_POS[0], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower0 = np.array(mean_curves_B_PRE[0]) - np.array(std_dev_curves_B_PRE[0])
        upper0 = np.array(mean_curves_B_PRE[0]) + np.array(std_dev_curves_B_PRE[0])
        ax1.fill_between(np.arange(len(mean_curves_B_PRE[0])), lower0, upper0, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower1 = np.array(mean_curves_B_POS[0]) - np.array(std_dev_curves_B_POS[0])
        upper1 = np.array(mean_curves_B_POS[0]) + np.array(std_dev_curves_B_POS[0])
        ax1.fill_between(np.arange(len(mean_curves_B_POS[0])), lower1, upper1, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax1.set_ylabel('Position', fontsize=28)
        ax1.set_xlabel('Samples', fontsize=28)
        ax1.text(0.5, 1.085,  f'Right Hand X', fontsize=28, ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 1.22, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=28)
        ax1.tick_params(axis='y', labelsize=28)
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax1.set_ylim([min_value_X-0.06, max_value_X+0.08])

        ax2.plot(np.arange(len(mean_curves_B_PRE[1])), mean_curves_B_PRE[1], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax2.plot(np.arange(len(mean_curves_B_POS[1])), mean_curves_B_POS[1], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower2 = np.array(mean_curves_B_PRE[1]) - np.array(std_dev_curves_B_PRE[1])
        upper2 = np.array(mean_curves_B_PRE[1]) + np.array(std_dev_curves_B_PRE[1])
        ax2.fill_between(np.arange(len(mean_curves_B_PRE[1])), lower2, upper2, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower3 = np.array(mean_curves_B_POS[1]) - np.array(std_dev_curves_B_POS[1])
        upper3 = np.array(mean_curves_B_POS[1]) + np.array(std_dev_curves_B_POS[1])
        ax2.fill_between(np.arange(len(mean_curves_B_POS[1])), lower3, upper3, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax2.set_ylabel('Position', fontsize=28)
        ax2.set_xlabel('Samples', fontsize=28)
        ax2.text(0.5, 1.085,  f'Right Hand Y', fontsize=28, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 1.22, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(axis='x', labelsize=28)
        ax2.tick_params(axis='y', labelsize=28)
        ax2.set_ylim([min_value_Y-0.06, max_value_Y+0.08])

        ax3.plot(np.arange(len(mean_curves_B_PRE[2])), mean_curves_B_PRE[2], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax3.plot(np.arange(len(mean_curves_B_POS[2])), mean_curves_B_POS[2], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower4 = np.array(mean_curves_B_PRE[2]) - np.array(std_dev_curves_B_PRE[2])
        upper4 = np.array(mean_curves_B_PRE[2]) + np.array(std_dev_curves_B_PRE[2])
        ax3.fill_between(np.arange(len(mean_curves_B_PRE[2])), lower4, upper4, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower5 = np.array(mean_curves_B_POS[2]) - np.array(std_dev_curves_B_POS[2])
        upper5 = np.array(mean_curves_B_POS[2]) + np.array(std_dev_curves_B_POS[2])
        ax3.fill_between(np.arange(len(mean_curves_B_POS[2])), lower5, upper5, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax3.set_ylabel('Position', fontsize=28)
        ax3.set_xlabel('Samples', fontsize=28)
        ax3.text(0.5, 1.085,  f'Right Hand Z', fontsize=28, ha='center', va='center', transform=ax3.transAxes)
        ax3.text(0.5, 1.22, 'Session C', fontsize=28, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True)
        ax3.tick_params(axis='x', labelsize=28)
        ax3.tick_params(axis='y', labelsize=28)
        ax3.set_ylim([min_value_Z-0.06, max_value_Z+0.08])

        ax4.plot(np.arange(len(mean_curves_B_PRE[3])), mean_curves_B_PRE[3], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax4.plot(np.arange(len(mean_curves_B_POS[3])), mean_curves_B_POS[3], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower6 = np.array(mean_curves_B_PRE[3]) - np.array(std_dev_curves_B_PRE[3])
        upper6 = np.array(mean_curves_B_PRE[3]) + np.array(std_dev_curves_B_PRE[3])
        ax4.fill_between(np.arange(len(mean_curves_B_PRE[3])), lower6, upper6, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower7 = np.array(mean_curves_B_POS[3]) - np.array(std_dev_curves_B_POS[3])
        upper7 = np.array(mean_curves_B_POS[3]) + np.array(std_dev_curves_B_POS[3])
        ax4.fill_between(np.arange(len(mean_curves_B_POS[3])), lower7, upper7, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax4.set_ylabel('Position', fontsize=28)
        ax4.set_xlabel('Samples', fontsize=28)
        ax4.text(0.5, 1.085,  f'Left Hand X', fontsize=28, ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True)
        ax4.tick_params(axis='x', labelsize=28)
        ax4.tick_params(axis='y', labelsize=28)
        ax4.set_ylim([min_value_X-0.06, max_value_X+0.08])

        ax5.plot(np.arange(len(mean_curves_B_PRE[4])), mean_curves_B_PRE[4], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax5.plot(np.arange(len(mean_curves_B_POS[4])), mean_curves_B_POS[4], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower8 = np.array(mean_curves_B_PRE[4]) - np.array(std_dev_curves_B_PRE[4])
        upper8 = np.array(mean_curves_B_PRE[4]) + np.array(std_dev_curves_B_PRE[4])
        ax5.fill_between(np.arange(len(mean_curves_B_PRE[4])), lower8, upper8, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower9 = np.array(mean_curves_B_POS[4]) - np.array(std_dev_curves_B_POS[4])
        upper9 = np.array(mean_curves_B_POS[4]) + np.array(std_dev_curves_B_POS[4])
        ax5.fill_between(np.arange(len(mean_curves_B_POS[4])), lower9, upper9, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax5.set_ylabel('Position', fontsize=28)
        ax5.set_xlabel('Samples', fontsize=28)
        ax5.text(0.5, 1.085,  f'Left Hand Y', fontsize=28, ha='center', va='center', transform=ax5.transAxes)
        ax5.grid(True)
        ax5.tick_params(axis='x', labelsize=28)
        ax5.tick_params(axis='y', labelsize=28)
        ax5.set_ylim([min_value_Y-0.06, max_value_Y+0.08])

        ax6.plot(np.arange(len(mean_curves_B_PRE[5])), mean_curves_B_PRE[5], '-', label=f'Mean Pre-VRMT', color=plt.cm.Reds(0.7), linewidth=2.0)
        ax6.plot(np.arange(len(mean_curves_B_POS[5])), mean_curves_B_POS[5], '-', label=f'Mean Post-VRMT', color=plt.cm.Greens(0.85), linewidth=2.0)
        lower10 = np.array(mean_curves_B_PRE[5]) - np.array(std_dev_curves_B_PRE[5])
        upper10 = np.array(mean_curves_B_PRE[5]) + np.array(std_dev_curves_B_PRE[5])
        ax6.fill_between(np.arange(len(mean_curves_B_PRE[5])), lower10, upper10, alpha=0.3, edgecolor=plt.cm.Reds(0.9), facecolor=plt.cm.Reds(0.6), label='Std. Pre-VRMT')
        lower11 = np.array(mean_curves_B_POS[5]) - np.array(std_dev_curves_B_POS[5])
        upper11 = np.array(mean_curves_B_POS[5]) + np.array(std_dev_curves_B_POS[5])
        ax6.fill_between(np.arange(len(mean_curves_B_POS[5])), lower11, upper11, alpha=0.3, edgecolor=plt.cm.BuGn(0.9), facecolor=plt.cm.BuGn(0.6), label='Std. Post-VRMT')
        ax6.set_ylabel('Position', fontsize=28)
        ax6.set_xlabel('Samples', fontsize=28)
        ax6.text(0.5, 1.085,  f'Left Hand Z', fontsize=28, ha='center', va='center', transform=ax6.transAxes)
        ax6.grid(True)
        ax6.tick_params(axis='x', labelsize=28)
        ax6.tick_params(axis='y', labelsize=28)
        ax6.set_ylim([min_value_Z-0.06, max_value_Z+0.08])

        left = 0.035
        right = 0.99
        hspace = 0.3
        wspace=0.25
        top = 0.95
        bottom = 0.08

    savefig = f"Parkinson/participant_{participant+9}/final_plots/final_act_{activity}.png"
    savefig_dissertation = os.path.join('..', 'Dissertation/images/final_plots', f'PD_final_act_1.png')

    plt.legend(handles, labels, bbox_to_anchor=(-0.15, -0.48), loc="lower center", ncol=4, fontsize=25)
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    plt.savefig(savefig)
    plt.savefig(savefig_dissertation)
    #plt.show()

############################### GET DATA PACKETS ###############################

def create_healthy_data_packets(number_participants_healthy: int):
    """
    Creates data packets for healthy participants by reading CSV files from specific directories.

    Args:
        number_participants_healthy (int): The number of healthy participants to process.

    Returns:
        list: A list of tuples containing DataFrames for each participant and session.
    """
    df = []

    for i in range(number_participants_healthy):
            
        df1A_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity1/Activity1-PRE.csv", delimiter=';')
        if not df1A_PRE.empty:
            df1A_PRE = df1A_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df1A_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity1/Activity1-POS.csv", delimiter=';')
        if not df1A_POS.empty:
            df1A_POS = df1A_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df1B_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity1/Activity1-PRE.csv", delimiter=';')
        if not df1B_PRE.empty:
            df1B_PRE = df1B_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df1B_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity1/Activity1-POS.csv", delimiter=';')
        if not df1B_POS.empty:
            df1B_POS = df1B_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df1C_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity1/Activity1-PRE.csv", delimiter=';')
        if not df1C_PRE.empty:
            df1C_PRE = df1C_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df1C_DURING = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity1/Activity1-DURING.csv", delimiter=';')
        if not df1C_DURING.empty:
            df1C_DURING = df1C_DURING.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df1C_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity1/Activity1-POS.csv", delimiter=';')
        if not df1C_POS.empty:
            df1C_POS = df1C_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df2A_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity2/backup/Activity2-PRE.csv", delimiter=';')
        if not df2A_PRE.empty:
            df2A_PRE = df2A_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df2A_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity2/backup/Activity2-POS.csv", delimiter=';')
        if not df2A_POS.empty:
            df2A_POS = df2A_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df2B_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity2/backup/Activity2-PRE.csv", delimiter=';')
        if not df2B_PRE.empty:
            df2B_PRE = df2B_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df2B_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity2/backup/Activity2-POS.csv", delimiter=';')
        if not df2B_POS.empty:
            df2B_POS = df2B_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df2C_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity2/backup/Activity2-PRE.csv", delimiter=';')
        if not df2C_PRE.empty:
            df2C_PRE = df2C_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df2C_DURING = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity2/Activity2-DURING.csv", delimiter=';')
        if not df2C_DURING.empty:
            df2C_DURING = df2C_DURING.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df2C_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity2/backup/Activity2-POS.csv", delimiter=';')
        if not df2C_POS.empty:
            df2C_POS = df2C_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df31A_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity31/Activity31-PRE.csv", delimiter=';')
        if not df31A_PRE.empty:
            df31A_PRE = df31A_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df31A_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity31/Activity31-POS.csv", delimiter=';')
        if not df31A_POS.empty:
            df31A_POS = df31A_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df31B_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity31/Activity31-PRE.csv", delimiter=';')
        if not df31B_PRE.empty:
            df31B_PRE = df31B_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df31B_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity31/Activity31-POS.csv", delimiter=';')
        if not df31B_POS.empty:
            df31B_POS = df31B_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df31C_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity31/Activity31-PRE.csv", delimiter=';')
        if not df31C_PRE.empty:
            df31C_PRE = df31C_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df31C_DURING = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity31/Activity31-DURING.csv", delimiter=';')
        if not df31C_DURING.empty:
            df31C_DURING = df31C_DURING.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df31C_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity31/Activity31-POS.csv", delimiter=';')
        if not df31C_POS.empty:
            df31C_POS = df31C_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df32A_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity32/Activity32-PRE.csv", delimiter=';')
        if not df32A_PRE.empty:
            df32A_PRE = df32A_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df32A_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_a/Activity32/Activity32-POS.csv", delimiter=';')
        if not df32A_POS.empty:
            df32A_POS = df32A_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df32B_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity32/Activity32-PRE.csv", delimiter=';')
        if not df32B_PRE.empty:
            df32B_PRE = df32B_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df32B_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_b/Activity32/Activity32-POS.csv", delimiter=';')
        if not df32B_POS.empty:
            df32B_POS = df32B_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df32C_PRE = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity32/Activity32-PRE.csv", delimiter=';')
        if not df32C_PRE.empty:
            df32C_PRE = df32C_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df32C_DURING = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity32/Activity32-DURING.csv", delimiter=';')
        if not df32C_DURING.empty:
            df32C_DURING = df32C_DURING.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df32C_POS = pd.read_csv(f"Healthy/participant_{i+1}/session_c/Activity32/Activity32-POS.csv", delimiter=';')
        if not df32C_POS.empty:
            df32C_POS = df32C_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)

        df.append(((df1A_PRE, df1A_POS, df2A_PRE, df2A_POS, df31A_PRE, df31A_POS, df32A_PRE, df32A_POS), (df1B_PRE, df1B_POS, df2B_PRE, df2B_POS, df31B_PRE, df31B_POS, df32B_PRE, df32B_POS), (df1C_PRE, df1C_DURING, df1C_POS, df2C_PRE, df2C_DURING, df2C_POS, df31C_PRE, df31C_DURING, df31C_POS, df32C_PRE, df32C_DURING, df32C_POS)))  
    
    return df

def create_PD_data_packets(number_participants_PD, PD_activity):
    """
    Creates data packets for PD participants by reading CSV files from specific directories.

    Args:
        number_participants_PD (int): The number of PD participants to process.

    Returns:
        list: A list of tuples containing DataFrames for each participant and session.
    """
    df = []
    for i in range(number_participants_PD):
        dfB_PRE = pd.read_csv(f"Parkinson/participant_{i+9}/session_c/Activity{PD_activity[i]}/Activity{PD_activity[i]}-PRE.csv", delimiter=';')
        dfB_PRE = dfB_PRE.apply(lambda x: x.str.replace(',', '.')).astype(float)
        dfB_POS = pd.read_csv(f"Parkinson/participant_{i+9}/session_c/Activity{PD_activity[i]}/Activity{PD_activity[i]}-POS.csv", delimiter=';')
        dfB_POS = dfB_POS.apply(lambda x: x.str.replace(',', '.')).astype(float)
        dfB_DURING = pd.read_csv(f"Parkinson/participant_{i+9}/session_c/Activity{PD_activity[i]}/Activity{PD_activity[i]}-DURING.csv", delimiter=';')
        dfB_DURING = dfB_DURING.apply(lambda x: x.str.replace(',', '.')).astype(float)
        df.append((dfB_PRE, dfB_POS, dfB_DURING))  
    return df

############################### EXTRACT FEATURES ###############################

def write_activity_features(file, headers, mean_curves, std_curves, activity_type):
    """
    Writes the features of the specified activity type in a .txt file.

    Args:
        file: The file object to write to.
        headers: The headers for the activities.
        mean_curves: The mean curves for the activities.
        std_curves: The standard deviation curves for the activities.
        activity_type: The type of activity (activity number and evaluation time point).
        is_position_activity: Boolean indicating if the activity is position-based.
    """
    file.write(f"Activity {activity_type}\n\n")
    for m, mean_curve in enumerate(mean_curves):
        file.write(f"{headers[m]}:\n\n")
        max_value = np.nanmax(mean_curve)
        min_value = np.nanmin(mean_curve)
        std_max = std_curves[m][np.nanargmax(mean_curve)]
        std_min = std_curves[m][np.nanargmin(mean_curve)]

        if ("1" in activity_type and "3" not in activity_type) or ("32" in activity_type and "Velocity" in headers[m]):
            file.write(f"Max: {max_value}\nMin: {min_value}\nStd Max: {std_max}\nStd Min: {std_min}\n\n")
        else:
            amplitude = max_value - min_value
            file.write(f"Amplitude: {amplitude}\nStd: {std_max}\n\n\n")

def calculate_difference(file, headers, mean_curves_PRE, mean_curves_POS, activity: int):
    """
    Calculates and writes in a .txt file the differences between PRE and POS, according to a specific activity.

    Args:
        file (str): The file object to write to.
        headers (list): The headers for the activities.
        mean_curves_PRE (list): The mean curves for the PRE activities.
        mean_curves_POS (list): The mean curves for the POS activities.
        activity_type (int): The type of activity ('1', '2', '31', '32').
    """
    file.write(f"Activity {str(activity)}\n\n")
    
    for m, mean_curve_PRE in enumerate(mean_curves_PRE):
        file.write(f"{headers[m]}:\n\n")
        
        max_value_PRE = np.nanmax(mean_curve_PRE)
        min_value_PRE = np.nanmin(mean_curve_PRE)
        max_value_POS = np.nanmax(mean_curves_POS[m])
        min_value_POS = np.nanmin(mean_curves_POS[m])
        
        if activity == 1 or (activity == 32 and "Velocity" in headers[m]):
            difference_max = max_value_POS - max_value_PRE
            difference_min = min_value_POS - min_value_PRE
            file.write(f"[MAX] Difference: {difference_max:.10f}".replace('.', ',') + "\n")
            file.write(f"[MIN] Difference: {difference_min:.10f}".replace('.', ',') + "\n\n")
        else:
            amplitude_PRE = np.nanmax(mean_curve_PRE) - np.nanmin(mean_curve_PRE)
            amplitude_POS = np.nanmax(mean_curves_POS[m]) - np.nanmin(mean_curves_POS[m])
            difference = amplitude_PRE - amplitude_POS
            file.write(f"Difference: {difference:.10f}".replace('.', ',') + "\n\n")

def find_healthy_features(session: str, participant:int, all_mean_1_PRE, all_std_1_PRE, all_mean_1_POS, all_std_1_POS, all_mean_2_PRE, all_std_2_PRE, all_mean_2_POS, all_std_2_POS, all_mean_31_PRE, all_std_31_PRE, all_mean_31_POS, all_std_31_POS, all_mean_32_PRE, all_std_32_PRE, all_mean_32_POS, all_std_32_POS):
    """
    Extracts and writes healthy features to .txt files for a specified session and participant.

    Args:
        session (str): The session identifier ('A', 'B', 'C').
        participant (int): The participant number (0-indexed, -1 for all participants).
        all_mean_1_PRE, all_std_1_PRE, all_mean_1_POS, all_std_1_POS: Feature data for Activity 1.
        all_mean_2_PRE, all_std_2_PRE, all_mean_2_POS, all_std_2_POS: Feature data for Activity 2.
        all_mean_31_PRE, all_std_31_PRE, all_mean_31_POS, all_std_31_POS: Feature data for Activity 31.
        all_mean_32_PRE, all_std_32_PRE, all_mean_32_POS, all_std_32_POS: Feature data for Activity 32.
    """
    participant_path = f'/participant_{participant + 1}' if participant != -1 else ''
    
    file_name = f"Healthy{participant_path}/features_healthy_session_{session}.txt"
    file_name_2 = f"Healthy{participant_path}/differences_healthy_session_{session}.txt" if participant != -1 else None

    with open(file_name, 'w') as file:
        write_activity_features(file, activity1_headers, all_mean_1_PRE, all_std_1_PRE, '1-PRE')
        write_activity_features(file, activity1_headers, all_mean_1_POS, all_std_1_POS, '1-POS')
        file.write(f"-----------------------------------\n\n")
        write_activity_features(file, activity2_headers, all_mean_2_PRE, all_std_2_PRE, '2-PRE')
        write_activity_features(file, activity2_headers, all_mean_2_POS, all_std_2_POS, '2-POS')
        file.write(f"-----------------------------------\n\n")
        write_activity_features(file, activity31_headers, all_mean_31_PRE, all_std_31_PRE, '31-PRE')
        write_activity_features(file, activity31_headers, all_mean_31_POS, all_std_31_POS, '31-POS')
        file.write(f"-----------------------------------\n\n")
        write_activity_features(file, activity32_headers, all_mean_32_PRE, all_std_32_PRE, '32-PRE')
        write_activity_features(file, activity32_headers, all_mean_32_POS, all_std_32_POS, '32-POS')

    if file_name_2:
        with open(file_name_2, 'w') as file2:
            calculate_difference(file2, activity1_headers, all_mean_1_PRE, all_mean_1_POS, 1)
            file2.write(f"-----------------------------------\n\n")
            calculate_difference(file2, activity2_headers, all_mean_2_PRE, all_mean_2_POS, 2)
            file2.write(f"-----------------------------------\n\n")
            calculate_difference(file2, activity31_headers, all_mean_31_PRE, all_mean_31_POS, 31)
            file2.write(f"-----------------------------------\n\n")
            calculate_difference(file2, activity32_headers, all_mean_32_PRE, all_mean_32_POS, 32)

def find_PD_features(number_participants, activity, all_mean_curves_B_PRE, all_std_dev_curves_B_PRE, all_mean_curves_B_POS, all_std_dev_curves_B_POS, all_mean_curves_B_DURING, all_std_dev_curves_B_DURING):
    """
    Extracts and writes PD features to .txt files for a specified session and participant.

    Args:
        number_participants (int): Number of PD participants.
        activity (int): Activity identifier.
        all_mean_curves_B_PRE, all_std_dev_curves_B_PRE, all_mean_curves_B_POS, all_std_dev_curves_B_POS, all_mean_curves_B_DURING, all_std_dev_curves_B_DURING: Feature data for Activity 1.
    """
    activity_params = {
        1: activity1_headers,
        2: activity2_headers,
        32: activity32_headers,
        31: activity31_headers
    }
    for participant_id in range(number_participants):
        headers = activity_params.get(activity[participant_id], None)
        with open(f'Parkinson/features_PD_Participant{participant_id+9}.txt', 'w') as file:
            write_activity_features(file, headers, all_mean_curves_B_PRE, all_std_dev_curves_B_PRE, "1-PRE")
            write_activity_features(file, headers, all_mean_curves_B_POS, all_std_dev_curves_B_POS, "1-POST")
            write_activity_features(file, headers, all_mean_curves_B_DURING, all_std_dev_curves_B_DURING, "1-DURING")

############################### HIGH-PASS FILTER ###############################

def apply_high_pass_filter_second_order(signal):
    Q = 2.0
    omega = 2 * math.pi * 3.0 / 58.8235
    alpha = round(math.sin(omega) / (2 * Q), 5)

    a0 = 1 + alpha
    a1 = -2 * round(math.cos(omega), 5)
    a2 = 1 - alpha
    b0 = (1 + round(math.cos(omega), 5)) / 2
    b1 = -(1 + round(math.cos(omega), 5))
    b2 = (1 + round(math.cos(omega), 5)) / 2

    filtered = np.zeros_like(signal, dtype=float)
    for i in range(2, len(signal)):
        filtered[i] = (b0 / a0) * signal[i] + (b1 / a0) * signal[i - 1] + (b2 / a0) * signal[i - 2] \
                      - (a1 / a0) * filtered[i - 1] - (a2 / a0) * filtered[i - 2]
    
    return filtered

############################### SUPLEMENTARY FUNCTION ###############################

def find_max_value(max_samples_per_participant):
    max_value_A_1, max_value_A_2, max_value_A_31, max_value_A_32 = 0, 0, 0, 0
    max_value_B_1, max_value_B_2, max_value_B_31, max_value_B_32 = 0, 0, 0, 0
    max_value_C_1, max_value_C_2, max_value_C_31, max_value_C_32 = 0, 0, 0, 0

    for i in range(number_healthy):
        current_max_A_1 = max_samples_per_participant[i][0][0]
        max_value_A_1 = max(max_value_A_1, current_max_A_1)
        current_max_A_2 = max_samples_per_participant[i][0][1]
        max_value_A_2 = max(max_value_A_2, current_max_A_2)
        current_max_A_31 = max_samples_per_participant[i][0][2]
        max_value_A_31 = max(max_value_A_31, current_max_A_31)
        current_max_A_32 = max_samples_per_participant[i][0][3]
        max_value_A_32 = max(max_value_A_32, current_max_A_32)

        current_max_B_1 = max_samples_per_participant[i][1][0]
        max_value_B_1 = max(max_value_B_1, current_max_B_1)
        current_max_B_2 = max_samples_per_participant[i][1][1]
        max_value_B_2 = max(max_value_B_2, current_max_B_2)
        current_max_B_31 = max_samples_per_participant[i][1][2]
        max_value_B_31 = max(max_value_B_31, current_max_B_31)
        current_max_B_32 = max_samples_per_participant[i][1][3]
        max_value_B_32 = max(max_value_B_32, current_max_B_32)

        current_max_C_1 = max_samples_per_participant[i][2][0]
        max_value_C_1 = max(max_value_C_1, current_max_C_1)
        current_max_C_2 = max_samples_per_participant[i][2][1]
        max_value_C_2 = max(max_value_C_2, current_max_C_2)
        current_max_C_31 = max_samples_per_participant[i][2][2]
        max_value_C_31 = max(max_value_C_31, current_max_C_31)
        current_max_C_32 = max_samples_per_participant[i][2][3]
        max_value_C_32 = max(max_value_C_32, current_max_C_32)
    return (max_value_A_1, max_value_A_2, max_value_A_31, max_value_A_32), (max_value_B_1, max_value_B_2, max_value_B_31, max_value_B_32), (max_value_C_1, max_value_C_2, max_value_C_31, max_value_C_32)

def create_healthy_mean_curves_by_activity(df, number_participants_healthy):
    mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS = [], [], [], [], [], [], [], []
    mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS = [], [], [], [], [], [], [], []
    mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS = [], [], [], [], [], [], [], [], [], [], [], []

    max_samples_per_participant = [
        [(0,0,0,0), (0,0,0,0), (0,0,0,0)]  # for each participant
        for _ in range(number_participants_healthy)  # for all participants
    ]

    for i in range(number_participants_healthy):

        segments_1A_PRE, max_1A_PRE = segment_activity(df[i][0][0], i+1, 1, activity1_indexes_per_healthy_A_PRE)
        segments_1A_POS, max_1A_POS = segment_activity(df[i][0][1], i+1, 1, activity1_indexes_per_healthy_A_POS)
        #plot_data(segments_1A_PRE, i+1, 1, "A", "PRE", True, "Segmentation")
        #plot_data(segments_1A_POS, i+1, 1, "A", "POS", True, "Segmentation")
        segments_2A_PRE, max_2A_PRE = segment_activity(df[i][0][2], i+1, 2, activity2_indexes_per_healthy_A_PRE)
        segments_2A_POS, max_2A_POS = segment_activity(df[i][0][3], i+1, 2, activity2_indexes_per_healthy_A_POS)
        #plot_data(segments_2A_PRE, i+1, 2, "A", "PRE", True, "Segmentation")
        #plot_data(segments_2A_POS, i+1, 2, "A", "POS", True, "Segmentation")
        segments_31A_PRE, max_31A_PRE = segment_activity(df[i][0][4], i+1, 31, activity3_rest_indexes_per_healthy_A_PRE)
        segments_31A_POS, max_31A_POS = segment_activity(df[i][0][5], i+1, 31, activity3_rest_indexes_per_healthy_A_POS)
        #plot_data(segments_31A_PRE, i+1, 31, "A", "PRE", True, "Segmentation")
        #plot_data(segments_31A_POS, i+1, 31, "A", "POS", True, "Segmentation")
        segments_32A_PRE, max_32A_PRE = segment_activity(df[i][0][6], i+1, 32, activity3_indexes_per_healthy_A_PRE)
        segments_32A_POS, max_32A_POS = segment_activity(df[i][0][7], i+1, 32, activity3_indexes_per_healthy_A_POS)
        #plot_data(segments_32A_PRE, i+1, 32, "A", "PRE", True, "Segmentation")
        #plot_data(segments_32A_POS, i+1, 32, "A", "POS", True, "Segmentation")

        max_samples_per_participant[i][0] = (max(max_1A_PRE, max_1A_POS), max(max_2A_PRE, max_2A_POS), max(max_31A_PRE, max_31A_POS), max(max_32A_PRE, max_32A_POS))

        segments_1B_PRE, max_1B_PRE = segment_activity(df[i][1][0], i+1, 1, activity1_indexes_per_healthy_B_PRE)
        segments_1B_POS, max_1B_POS = segment_activity(df[i][1][1], i+1, 1, activity1_indexes_per_healthy_B_POS)
        #plot_data(segments_1B_PRE, i+1, 1, "B", "PRE", True, "Segmentation")
        #plot_data(segments_1B_POS, i+1, 1, "B", "POS", True, "Segmentation")
        segments_2B_PRE, max_2B_PRE = segment_activity(df[i][1][2], i+1, 2, activity2_indexes_per_healthy_B_PRE)
        segments_2B_POS, max_2B_POS = segment_activity(df[i][1][3], i+1, 2, activity2_indexes_per_healthy_B_POS)
        #plot_data(segments_2B_PRE, i+1, 2, "B", "PRE", True, "Segmentation")
        #plot_data(segments_2B_POS, i+1, 2, "B", "POS", True, "Segmentation")
        segments_31B_PRE, max_31B_PRE = segment_activity(df[i][1][4], i+1, 31, activity3_rest_indexes_per_healthy_B_PRE)
        segments_31B_POS, max_31B_POS = segment_activity(df[i][1][5], i+1, 31, activity3_rest_indexes_per_healthy_B_POS)
        #plot_data(segments_31B_PRE, i+1, 31, "B", "PRE", True, "Segmentation")
        #plot_data(segments_31B_POS, i+1, 31, "B", "POS", True, "Segmentation")
        segments_32B_PRE, max_32B_PRE = segment_activity(df[i][1][6], i+1, 32, activity3_indexes_per_healthy_B_PRE)
        segments_32B_POS, max_32B_POS = segment_activity(df[i][1][7], i+1, 32, activity3_indexes_per_healthy_B_POS)
        #plot_data(segments_32B_PRE, i+1, 32, "B", "PRE", True, "Segmentation")
        #plot_data(segments_32B_POS, i+1, 32, "B", "POS", True, "Segmentation")

        max_samples_per_participant[i][1] = (max(max_1B_PRE, max_1B_POS), max(max_2B_PRE, max_2B_POS), max(max_31B_PRE, max_31B_POS), max(max_32B_PRE, max_32B_POS))

        segments_1C_PRE, max_1C_PRE = segment_activity(df[i][2][0], i+1, 1, activity1_indexes_per_healthy_C_PRE)
        segments_1C_DURING, max_1C_DURING = segment_activity(df[i][2][1], i+1, 1, activity1_indexes_per_healthy_C_DURING)
        segments_1C_POS, max_1C_POS = segment_activity(df[i][2][2], i+1, 1, activity1_indexes_per_healthy_C_POS)
        #plot_data(segments_1C_PRE, i+1, 1, "C", "PRE", True, "Segmentation")
        #plot_data(segments_1C_DURING, i+1, 1, "C", "DURING", True, "Segmentation")
        #plot_data(segments_1C_POS, i+1, 1, "C", "POS", True, "Segmentation")
        segments_2C_PRE, max_2C_PRE = segment_activity(df[i][2][3], i+1, 2, activity2_indexes_per_healthy_C_PRE)
        segments_2C_DURING, max_2C_DURING = segment_activity(df[i][2][4], i+1, 2, activity2_indexes_per_healthy_C_DURING)
        segments_2C_POS, max_2C_POS = segment_activity(df[i][2][5], i+1, 2, activity2_indexes_per_healthy_C_POS)
        #plot_data(segments_2C_PRE, i+1, 2, "C", "PRE", True, "Segmentation")
        #plot_data(segments_2C_DURING, i+1, 2, "C", "DURING", True, "Segmentation")
        #plot_data(segments_2C_POS, i+1, 2, "C", "POS", True, "Segmentation")
        segments_31C_PRE, max_31C_PRE = segment_activity(df[i][2][6], i+1, 31, activity3_rest_indexes_per_healthy_C_PRE)
        segments_31C_DURING, max_31C_DURING = segment_activity(df[i][2][7], i+1, 31, activity3_rest_indexes_per_healthy_C_DURING)
        segments_31C_POS, max_31C_POS = segment_activity(df[i][2][8], i+1, 31, activity3_rest_indexes_per_healthy_C_POS)
        #plot_data(segments_31C_PRE, i+1, 31, "C", "PRE", True, "Segmentation")
        #plot_data(segments_31C_DURING, i+1, 31, "C", "DURING", True, "Segmentation")
        #plot_data(segments_31C_POS, i+1, 31, "C", "POS", True, "Segmentation")
        segments_32C_PRE, max_32C_PRE = segment_activity(df[i][2][9], i+1, 32, activity3_indexes_per_healthy_C_PRE)
        segments_32C_DURING, max_32C_DURING = segment_activity(df[i][2][10], i+1, 32, activity3_indexes_per_healthy_C_DURING)
        segments_32C_POS, max_32C_POS = segment_activity(df[i][2][11], i+1, 32, activity3_indexes_per_healthy_C_POS)
        #plot_data(segments_32C_PRE, i+1, 32, "C", "PRE", True, "Segmentation")
        #plot_data(segments_32C_DURING, i+1, 32, "C", "DURING", True, "Segmentation")
        #plot_data(segments_32C_POS, i+1, 32, "C", "POS", True, "Segmentation")

        #max_samples_per_participant[i][2] = (max(max_1C_PRE, max_1C_DURING, max_1C_POS), max(max_2C_PRE, max_2C_DURING, max_2C_POS), max(max_31C_PRE, max_31C_DURING, max_31C_POS), max(max_32C_PRE, max_32C_DURING, max_32C_POS))
        max_samples_per_participant[i][2] = (max(max_1C_PRE, max_1C_POS), max(max_2C_PRE, max_2C_POS), max(max_31C_PRE, max_31C_POS), max(max_32C_PRE, max_32C_POS))

        segments_1A_PRE_array, segments_1A_POS_array = convert_to_array(segments_1A_PRE, healthy_height[i]), convert_to_array(segments_1A_POS, healthy_height[i])
        segments_2A_PRE_array, segments_2A_POS_array = convert_to_array(segments_2A_PRE, healthy_height[i]), convert_to_array(segments_2A_POS, healthy_height[i])
        segments_31A_PRE_array, segments_31A_POS_array = convert_to_array(segments_31A_PRE, healthy_height[i]), convert_to_array(segments_31A_POS, healthy_height[i])
        segments_32A_PRE_array, segments_32A_POS_array = convert_to_array(segments_32A_PRE, healthy_height[i]), convert_to_array(segments_32A_POS, healthy_height[i])

        segments_1B_PRE_array, segments_1B_POS_array = convert_to_array(segments_1B_PRE, healthy_height[i]), convert_to_array(segments_1B_POS, healthy_height[i])
        segments_2B_PRE_array, segments_2B_POS_array = convert_to_array(segments_2B_PRE, healthy_height[i]), convert_to_array(segments_2B_POS, healthy_height[i])
        segments_31B_PRE_array, segments_31B_POS_array = convert_to_array(segments_31B_PRE, healthy_height[i]), convert_to_array(segments_31B_POS, healthy_height[i])
        segments_32B_PRE_array, segments_32B_POS_array = convert_to_array(segments_32B_PRE, healthy_height[i]), convert_to_array(segments_32B_POS, healthy_height[i])

        segments_1C_PRE_array, segments_1C_DURING_array, segments_1C_POS_array = convert_to_array(segments_1C_PRE, healthy_height[i]), convert_to_array(segments_1C_DURING, healthy_height[i]), convert_to_array(segments_1C_POS, healthy_height[i])
        segments_2C_PRE_array, segments_2C_DURING_array, segments_2C_POS_array = convert_to_array(segments_2C_PRE, healthy_height[i]), convert_to_array(segments_2C_DURING, healthy_height[i]), convert_to_array(segments_2C_POS, healthy_height[i])
        segments_31C_PRE_array, segments_31C_DURING_array, segments_31C_POS_array = convert_to_array(segments_31C_PRE, healthy_height[i]), convert_to_array(segments_31C_DURING, healthy_height[i]), convert_to_array(segments_31C_POS, healthy_height[i])
        segments_32C_PRE_array, segments_32C_DURING_array, segments_32C_POS_array = convert_to_array(segments_32C_PRE, healthy_height[i]), convert_to_array(segments_32C_DURING, healthy_height[i]), convert_to_array(segments_32C_POS, healthy_height[i])

        mean_1A_PRE_aux, std_1A_PRE_aux = calculate_mean_curve(segments_1A_PRE_array, max(max_samples_per_participant[i][0][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_1A_POS_aux, std_1A_POS_aux = calculate_mean_curve(segments_1A_POS_array, max(max_samples_per_participant[i][0][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_2A_PRE_aux, std_2A_PRE_aux = calculate_mean_curve(segments_2A_PRE_array, max(max_samples_per_participant[i][0][1],len(ACT2_COM_X)))
        mean_2A_POS_aux, std_2A_POS_aux = calculate_mean_curve(segments_2A_POS_array, max(max_samples_per_participant[i][0][1],len(ACT2_COM_X)))
        mean_31A_PRE_aux, std_31A_PRE_aux = calculate_mean_curve(segments_31A_PRE_array, max(max_samples_per_participant[i][0][2],len(ACT31_RIGHT_HAND_X)))
        mean_31A_POS_aux, std_31A_POS_aux = calculate_mean_curve(segments_31A_POS_array, max(max_samples_per_participant[i][0][2],len(ACT31_RIGHT_HAND_X)))
        mean_32A_PRE_aux, std_32A_PRE_aux = calculate_mean_curve(segments_32A_PRE_array, max(max_samples_per_participant[i][0][3],len(ACT32_RIGHT_HAND_POS)))
        mean_32A_POS_aux, std_32A_POS_aux = calculate_mean_curve(segments_32A_POS_array, max(max_samples_per_participant[i][0][3],len(ACT32_RIGHT_HAND_POS)))

        mean_1B_PRE_aux, std_1B_PRE_aux = calculate_mean_curve(segments_1B_PRE_array, max(max_samples_per_participant[i][1][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_1B_POS_aux, std_1B_POS_aux = calculate_mean_curve(segments_1B_POS_array, max(max_samples_per_participant[i][1][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_2B_PRE_aux, std_2B_PRE_aux = calculate_mean_curve(segments_2B_PRE_array, max(max_samples_per_participant[i][1][1],len(ACT2_COM_X)))
        mean_2B_POS_aux, std_2B_POS_aux = calculate_mean_curve(segments_2B_POS_array, max(max_samples_per_participant[i][1][1],len(ACT2_COM_X)))
        mean_31B_PRE_aux, std_31B_PRE_aux = calculate_mean_curve(segments_31B_PRE_array, max(max_samples_per_participant[i][1][2],len(ACT31_RIGHT_HAND_X)))
        mean_31B_POS_aux, std_31B_POS_aux = calculate_mean_curve(segments_31B_POS_array, max(max_samples_per_participant[i][1][2],len(ACT31_RIGHT_HAND_X)))
        mean_32B_PRE_aux, std_32B_PRE_aux = calculate_mean_curve(segments_32B_PRE_array, max(max_samples_per_participant[i][1][3],len(ACT32_RIGHT_HAND_POS)))
        mean_32B_POS_aux, std_32B_POS_aux = calculate_mean_curve(segments_32B_POS_array, max(max_samples_per_participant[i][1][3],len(ACT32_RIGHT_HAND_POS)))

        mean_1C_PRE_aux, std_1C_PRE_aux = calculate_mean_curve(segments_1C_PRE_array, max(max_samples_per_participant[i][2][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_1C_DURING_aux, std_1C_DURING_aux = calculate_mean_curve(segments_1C_DURING_array, max(max_samples_per_participant[i][2][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_1C_POS_aux, std_1C_POS_aux = calculate_mean_curve(segments_1C_POS_array, max(max_samples_per_participant[i][2][0],len(ACT1_RIGHT_FOOT_POS)))
        mean_2C_PRE_aux, std_2C_PRE_aux = calculate_mean_curve(segments_2C_PRE_array, max(max_samples_per_participant[i][2][1],len(ACT2_COM_X)))
        mean_2C_DURING_aux, std_2C_DURING_aux = calculate_mean_curve(segments_2C_DURING_array, max(max_samples_per_participant[i][2][1],len(ACT2_COM_X)))
        mean_2C_POS_aux, std_2C_POS_aux = calculate_mean_curve(segments_2C_POS_array, max(max_samples_per_participant[i][2][1],len(ACT2_COM_X)))
        mean_31C_PRE_aux, std_31C_PRE_aux = calculate_mean_curve(segments_31C_PRE_array, max(max_samples_per_participant[i][2][2],len(ACT31_RIGHT_HAND_X)))
        mean_31C_DURING_aux, std_31C_DURING_aux = calculate_mean_curve(segments_31C_DURING_array, max(max_samples_per_participant[i][2][2],len(ACT31_RIGHT_HAND_X)))
        mean_31C_POS_aux, std_31C_POS_aux = calculate_mean_curve(segments_31C_POS_array, max(max_samples_per_participant[i][2][2],len(ACT31_RIGHT_HAND_X)))
        mean_32C_PRE_aux, std_32C_PRE_aux = calculate_mean_curve(segments_32C_PRE_array, max(max_samples_per_participant[i][2][3],len(ACT32_RIGHT_HAND_POS)))
        mean_32C_DURING_aux, std_32C_DURING_aux = calculate_mean_curve(segments_32C_DURING_array, max(max_samples_per_participant[i][2][3],len(ACT32_RIGHT_HAND_POS)))
        mean_32C_POS_aux, std_32C_POS_aux = calculate_mean_curve(segments_32C_POS_array, max(max_samples_per_participant[i][2][3],len(ACT32_RIGHT_HAND_POS)))

        find_healthy_features("A", i, mean_1A_PRE_aux, std_1A_PRE_aux, mean_1A_POS_aux, std_1A_POS_aux, mean_2A_PRE_aux, std_2A_PRE_aux, mean_2A_POS_aux, std_2A_POS_aux, mean_31A_PRE_aux, std_31A_PRE_aux, mean_31A_POS_aux, std_31A_POS_aux, mean_32A_PRE_aux, std_32A_PRE_aux, mean_32A_POS_aux, std_32A_POS_aux)

        find_healthy_features("B", i, mean_1B_PRE_aux, std_1B_PRE_aux, mean_1B_POS_aux, std_1B_POS_aux, mean_2B_PRE_aux, std_2B_PRE_aux, mean_2B_POS_aux, std_2B_POS_aux, mean_31B_PRE_aux, std_31B_PRE_aux, mean_31B_POS_aux, std_31B_POS_aux, mean_32B_PRE_aux, std_32B_PRE_aux, mean_32B_POS_aux, std_32B_POS_aux)

        find_healthy_features("C", i, mean_1C_PRE_aux, std_1C_PRE_aux, mean_1C_POS_aux, std_1C_POS_aux, mean_2C_PRE_aux, std_2C_PRE_aux, mean_2C_POS_aux, std_2C_POS_aux, mean_31C_PRE_aux, std_31C_PRE_aux, mean_31C_POS_aux, std_31C_POS_aux, mean_32C_PRE_aux, std_32C_PRE_aux, mean_32C_POS_aux, std_32C_POS_aux)

        mean_1A_PRE.append(mean_1A_PRE_aux)
        mean_1A_POS.append(mean_1A_POS_aux)
        mean_2A_PRE.append(mean_2A_PRE_aux)
        mean_2A_POS.append(mean_2A_POS_aux)
        mean_31A_PRE.append(mean_31A_PRE_aux)
        mean_31A_POS.append(mean_31A_POS_aux)
        mean_32A_PRE.append(mean_32A_PRE_aux)
        mean_32A_POS.append(mean_32A_POS_aux)

        mean_1B_PRE.append(mean_1B_PRE_aux)
        mean_1B_POS.append(mean_1B_POS_aux)
        mean_2B_PRE.append(mean_2B_PRE_aux)
        mean_2B_POS.append(mean_2B_POS_aux)
        mean_31B_PRE.append(mean_31B_PRE_aux)
        mean_31B_POS.append(mean_31B_POS_aux)
        mean_32B_PRE.append(mean_32B_PRE_aux)
        mean_32B_POS.append(mean_32B_POS_aux)

        mean_1C_PRE.append(mean_1C_PRE_aux)
        mean_1C_DURING.append(mean_1C_DURING_aux)
        mean_1C_POS.append(mean_1C_POS_aux)
        mean_2C_PRE.append(mean_2C_PRE_aux)
        mean_2C_DURING.append(mean_2C_DURING_aux)
        mean_2C_POS.append(mean_2C_POS_aux)
        mean_31C_PRE.append(mean_31C_PRE_aux)
        mean_31C_DURING.append(mean_31C_DURING_aux)
        mean_31C_POS.append(mean_31C_POS_aux)
        mean_32C_PRE.append(mean_32C_PRE_aux)
        mean_32C_DURING.append(mean_32C_DURING_aux)
        mean_32C_POS.append(mean_32C_POS_aux)

        #plot_data(mean_1A_PRE_aux, i, 1, "A", "PRE", True, False, "Mean"), plot_data(mean_1A_POS_aux, i, 1, "A", "POS", True, False, "Mean"), plot_data(mean_2A_PRE_aux, i, 2, "A", "PRE", True, False, "Mean"), plot_data(mean_2A_POS_aux, i, 2, "A", "POS", True, False, "Mean"), plot_data(mean_31A_PRE_aux, i, 31, "A", "PRE", True, False, "Mean"), plot_data(mean_31A_POS_aux, i, 31, "A", "POS", True, False, "Mean"), plot_data(mean_32A_PRE_aux, i, 32, "A", "PRE", True, False, "Mean"), plot_data(mean_32A_POS_aux, i, 32, "A", "POS", True, False, "Mean")

        #plot_data(mean_1B_PRE_aux, i, 1, "B", "PRE", True, False, "Mean"), plot_data(mean_1B_POS_aux, i, 1, "B", "POS", True, False, "Mean"), plot_data(mean_2B_PRE_aux, i, 2, "B", "PRE", True, False, "Mean"), plot_data(mean_2B_POS_aux, i, 2, "B", "POS", True, False, "Mean"), plot_data(mean_31B_PRE_aux, i, 31, "B", "PRE", True, False, "Mean"), plot_data(mean_31B_POS_aux, i, 31, "B", "POS", True, False, "Mean"), plot_data(mean_32B_PRE_aux, i, 32, "B", "PRE", True, False, "Mean"), plot_data(mean_32B_POS_aux, i, 32, "B", "POS", True, False, "Mean")

        #plot_data(mean_1C_PRE_aux, i, 1, "C", "PRE", True, False, "Mean"), plot_data(mean_1C_POS_aux, i, 1, "C", "POS", True, False, "Mean"), plot_data(mean_2C_PRE_aux, i, 2, "C", "PRE", True, False, "Mean"), plot_data(mean_2C_POS_aux, i, 2, "C", "POS", True, False, "Mean"), plot_data(mean_31C_PRE_aux, i, 31, "C", "PRE", True, False, "Mean"), plot_data(mean_31C_POS_aux, i, 31, "C", "POS", True, False, "Mean"), plot_data(mean_32C_PRE_aux, i, 32, "C", "PRE", True, False, "Mean"), plot_data(mean_32C_POS_aux, i, 32, "C", "POS", True, False, "Mean")

        #plot_final_data_healthy(mean_1A_PRE_aux, std_1A_PRE_aux, mean_1A_POS_aux, std_1A_POS_aux, mean_1B_PRE_aux, std_1B_PRE_aux, mean_1B_POS_aux, std_1B_POS_aux, mean_1C_PRE_aux, std_1C_PRE_aux, mean_1C_DURING_aux, std_1C_DURING_aux, mean_1C_POS_aux, std_1C_POS_aux, i, 1, False)

        #plot_final_data_healthy(mean_2A_PRE_aux, std_2A_PRE_aux, mean_2A_POS_aux, std_2A_POS_aux, mean_2B_PRE_aux, std_2B_PRE_aux, mean_2B_POS_aux, std_2B_POS_aux, mean_2C_PRE_aux, std_2C_PRE_aux, mean_2C_DURING_aux, std_2C_DURING_aux, mean_2C_POS_aux, std_2C_POS_aux, i, 2, False)

        #plot_final_data_healthy(mean_31A_PRE_aux, std_31A_PRE_aux, mean_31A_POS_aux, std_31A_POS_aux, mean_31B_PRE_aux, std_31B_PRE_aux, mean_31B_POS_aux, std_31B_POS_aux, mean_31C_PRE_aux, std_31C_PRE_aux, mean_31C_DURING_aux, std_31C_DURING_aux, mean_31C_POS_aux, std_31C_POS_aux, i, 31, False)

        #plot_final_data_healthy(mean_32A_PRE_aux, std_32A_PRE_aux, mean_32A_POS_aux, std_32A_POS_aux, mean_32B_PRE_aux, std_32B_PRE_aux, mean_32B_POS_aux, std_32B_POS_aux, mean_32C_PRE_aux, std_32C_PRE_aux, mean_32C_DURING_aux, std_32C_DURING_aux, mean_32C_POS_aux, std_32C_POS_aux, i, 32, False)

    return max_samples_per_participant, mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS    

def create_PD_mean_curves_by_activity(df, number_participants_PD, PD_activity):
    mean_B_PRE, mean_B_POS, mean_B_DURING, std_B_PRE, std_B_POS, std_B_DURING = [], [], [], [], [], []
    max_samples_per_participant = [0]

    for i in range(number_participants_PD):

        segments_B_PRE, max_B_PRE = segment_activity(df[i][0], i+1, PD_activity[i], activity_indexes_per_PD_B_PRE)
        segments_B_POS, max_B_POS = segment_activity(df[i][1], i+1, PD_activity[i], activity_indexes_per_PD_B_POS)
        segments_B_DURING, max_B_DURING = segment_activity(df[i][2], i+1, PD_activity[i], activity_indexes_per_PD_B_DURING)
        max_samples_per_participant[i] = max(max_B_PRE, max_B_POS, max_B_DURING)
        plot_data(segments_B_PRE, i, PD_activity[i], "C", "PRE", False, "Segmentation")
        plot_data(segments_B_POS, i, PD_activity[i], "C", "POS", False, "Segmentation")
        plot_data(segments_B_DURING, i, PD_activity[i], "C", "DURING", False, "Segmentation")

        segments_B_PRE_array, segments_B_POS_array, segments_B_DURING_array = convert_to_array(segments_B_PRE, PD_height[i]), convert_to_array(segments_B_POS, PD_height[i]), convert_to_array(segments_B_DURING, PD_height[i])

        mean_B_PRE_aux, std_B_PRE_aux = calculate_mean_curve(segments_B_PRE_array, max_samples_per_participant[i])
        mean_B_POS_aux, std_B_POS_aux = calculate_mean_curve(segments_B_POS_array, max_samples_per_participant[i])
        mean_B_DURING_aux, std_B_DURING_aux = calculate_mean_curve(segments_B_DURING_array, max_samples_per_participant[i])

        mean_B_PRE.append(mean_B_PRE_aux)
        mean_B_POS.append(mean_B_POS_aux)
        std_B_PRE.append(std_B_PRE_aux)
        std_B_POS.append(std_B_POS_aux)
        mean_B_DURING.append(mean_B_DURING_aux)
        std_B_DURING.append(std_B_DURING_aux)

        plot_data(mean_B_PRE_aux, i, PD_activity[i], "C", "PRE", False, "Mean"), plot_data(mean_B_POS_aux, i, PD_activity[i], "C", "POS", False, "Mean"), plot_data(mean_B_DURING_aux, i, PD_activity[i], "C", "DURING", False, "Mean")

        plot_final_data_PD(mean_B_PRE_aux, std_B_PRE_aux, mean_B_POS_aux, std_B_POS_aux, mean_B_DURING_aux, std_B_DURING_aux, i, PD_activity[i])

    return mean_B_PRE_aux, mean_B_POS_aux, std_B_PRE_aux, std_B_POS_aux, mean_B_DURING_aux, std_B_DURING_aux

def group_mean_curves_by_healthy(mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS):
    all_healthy_mean_1A_PRE, all_healthy_mean_1A_POS, all_healthy_mean_2A_PRE, all_healthy_mean_2A_POS, all_healthy_mean_31A_PRE, all_healthy_mean_31A_POS, all_healthy_mean_32A_PRE, all_healthy_mean_32A_POS, all_healthy_mean_1B_PRE, all_healthy_mean_1B_POS, all_healthy_mean_2B_PRE, all_healthy_mean_2B_POS, all_healthy_mean_31B_PRE, all_healthy_mean_31B_POS, all_healthy_mean_32B_PRE, all_healthy_mean_32B_POS, all_healthy_mean_1C_PRE, all_healthy_mean_1C_DURING, all_healthy_mean_1C_POS, all_healthy_mean_2C_PRE, all_healthy_mean_2C_DURING, all_healthy_mean_2C_POS, all_healthy_mean_31C_PRE, all_healthy_mean_31C_DURING, all_healthy_mean_31C_POS, all_healthy_mean_32C_PRE, all_healthy_mean_32C_DURING, all_healthy_mean_32C_POS = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for k, parameter in enumerate(activity1_headers):
        participants_mean_curves = []
        for participant in mean_1A_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1A_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1A_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1A_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1B_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1B_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1B_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1B_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1C_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1C_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1C_DURING:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1C_DURING.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_1C_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_1C_POS.append(participants_mean_curves)
        
    for k, parameter in enumerate(activity2_headers):
        participants_mean_curves = []
        for participant in mean_2A_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2A_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2A_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2A_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2B_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2B_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2B_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2B_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2C_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2C_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2C_DURING:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2C_DURING.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_2C_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_2C_POS.append(participants_mean_curves)

    for k, parameter in enumerate(activity31_headers):
        participants_mean_curves = []
        for participant in mean_31A_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31A_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31A_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31A_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31B_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31B_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31B_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31B_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31C_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31C_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31C_DURING:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31C_DURING.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_31C_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_31C_POS.append(participants_mean_curves)

    for k, parameter in enumerate(activity32_headers):
        participants_mean_curves = []
        for participant in mean_32A_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32A_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32A_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32A_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32B_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32B_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32B_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32B_POS.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32C_PRE:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32C_PRE.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32C_DURING:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32C_DURING.append(participants_mean_curves)

        participants_mean_curves = []
        for participant in mean_32C_POS:
            participants_mean_curves.append(participant[k])
        all_healthy_mean_32C_POS.append(participants_mean_curves)

    return all_healthy_mean_1A_PRE, all_healthy_mean_1A_POS, all_healthy_mean_2A_PRE, all_healthy_mean_2A_POS, all_healthy_mean_31A_PRE, all_healthy_mean_31A_POS, all_healthy_mean_32A_PRE, all_healthy_mean_32A_POS, all_healthy_mean_1B_PRE, all_healthy_mean_1B_POS, all_healthy_mean_2B_PRE, all_healthy_mean_2B_POS, all_healthy_mean_31B_PRE, all_healthy_mean_31B_POS, all_healthy_mean_32B_PRE, all_healthy_mean_32B_POS, all_healthy_mean_1C_PRE, all_healthy_mean_1C_DURING, all_healthy_mean_1C_POS, all_healthy_mean_2C_PRE, all_healthy_mean_2C_DURING, all_healthy_mean_2C_POS, all_healthy_mean_31C_PRE, all_healthy_mean_31C_DURING, all_healthy_mean_31C_POS, all_healthy_mean_32C_PRE, all_healthy_mean_32C_DURING, all_healthy_mean_32C_POS

def find_all_healthy_mean_curve(max_samples_per_participant, mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS):

    all_mean_1A_PRE_aux, all_mean_1A_POS_aux, all_mean_2A_PRE_aux, all_mean_2A_POS_aux, all_mean_31A_PRE_aux, all_mean_31A_POS_aux, all_mean_32A_PRE_aux, all_mean_32A_POS_aux, all_mean_1B_PRE_aux, all_mean_1B_POS_aux, all_mean_2B_PRE_aux, all_mean_2B_POS_aux, all_mean_31B_PRE_aux, all_mean_31B_POS_aux, all_mean_32B_PRE_aux, all_mean_32B_POS_aux, all_mean_1C_PRE_aux, all_mean_1C_DURING_aux, all_mean_1C_POS_aux, all_mean_2C_PRE_aux, all_mean_2C_DURING_aux, all_mean_2C_POS_aux, all_mean_31C_PRE_aux, all_mean_31C_DURING_aux, all_mean_31C_POS_aux, all_mean_32C_PRE_aux, all_mean_32C_DURING_aux, all_mean_32C_POS_aux = group_mean_curves_by_healthy(mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS)

    max_values = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]

    max_tuple_A, max_tuple_B, max_tuple_C = find_max_value(max_samples_per_participant)
    max_values[0] = max_tuple_A
    max_values[1] = max_tuple_B
    max_values[2] = max_tuple_C

    all_mean_1A_PRE, all_std_dev_1A_PRE = calculate_mean_curve(all_mean_1A_PRE_aux, max_values[0][0])
    all_mean_1A_POS, all_std_dev_1A_POS = calculate_mean_curve(all_mean_1A_POS_aux, max_values[0][0])
    all_mean_2A_PRE, all_std_dev_2A_PRE = calculate_mean_curve(all_mean_2A_PRE_aux, max_values[0][1])
    all_mean_2A_POS, all_std_dev_2A_POS = calculate_mean_curve(all_mean_2A_POS_aux, max_values[0][1])
    all_mean_31A_PRE, all_std_dev_31A_PRE = calculate_mean_curve(all_mean_31A_PRE_aux, max_values[0][2])
    all_mean_31A_POS, all_std_dev_31A_POS = calculate_mean_curve(all_mean_31A_POS_aux, max_values[0][2])
    all_mean_32A_PRE, all_std_dev_32A_PRE = calculate_mean_curve(all_mean_32A_PRE_aux, max_values[0][3])
    all_mean_32A_POS, all_std_dev_32A_POS = calculate_mean_curve(all_mean_32A_POS_aux, max_values[0][3])

    all_mean_1B_PRE, all_std_dev_1B_PRE = calculate_mean_curve(all_mean_1B_PRE_aux, max_values[1][0])
    all_mean_1B_POS, all_std_dev_1B_POS = calculate_mean_curve(all_mean_1B_POS_aux, max_values[1][0])
    all_mean_2B_PRE, all_std_dev_2B_PRE = calculate_mean_curve(all_mean_2B_PRE_aux, max_values[1][1])
    all_mean_2B_POS, all_std_dev_2B_POS = calculate_mean_curve(all_mean_2B_POS_aux, max_values[1][1])
    all_mean_31B_PRE, all_std_dev_31B_PRE = calculate_mean_curve(all_mean_31B_PRE_aux, max_values[1][2])
    all_mean_31B_POS, all_std_dev_31B_POS = calculate_mean_curve(all_mean_31B_POS_aux, max_values[1][2])
    all_mean_32B_PRE, all_std_dev_32B_PRE = calculate_mean_curve(all_mean_32B_PRE_aux, max_values[1][3])
    all_mean_32B_POS, all_std_dev_32B_POS = calculate_mean_curve(all_mean_32B_POS_aux, max_values[1][3])

    all_mean_1C_PRE, all_std_dev_1C_PRE = calculate_mean_curve(all_mean_1C_PRE_aux, max_values[2][0])
    all_mean_1C_DURING, all_std_dev_1C_DURING = calculate_mean_curve(all_mean_1C_DURING_aux, max_values[2][0])
    all_mean_1C_POS, all_std_dev_1C_POS = calculate_mean_curve(all_mean_1C_POS_aux, max_values[2][0])
    all_mean_2C_PRE, all_std_dev_2C_PRE = calculate_mean_curve(all_mean_2C_PRE_aux, max_values[2][1])
    all_mean_2C_DURING, all_std_dev_2C_DURING = calculate_mean_curve(all_mean_2C_DURING_aux, max_values[2][1])
    all_mean_2C_POS, all_std_dev_2C_POS = calculate_mean_curve(all_mean_2C_POS_aux, max_values[2][1])
    all_mean_31C_PRE, all_std_dev_31C_PRE = calculate_mean_curve(all_mean_31C_PRE_aux, max_values[2][2])
    all_mean_31C_DURING, all_std_dev_31C_DURING = calculate_mean_curve(all_mean_31C_DURING_aux, max_values[2][2])
    all_mean_31C_POS, all_std_dev_31C_POS = calculate_mean_curve(all_mean_31C_POS_aux, max_values[2][2])
    all_mean_32C_PRE, all_std_dev_32C_PRE = calculate_mean_curve(all_mean_32C_PRE_aux, max_values[2][3])
    all_mean_32C_DURING, all_std_dev_32C_DURING = calculate_mean_curve(all_mean_32C_DURING_aux, max_values[2][3])
    all_mean_32C_POS, all_std_dev_32C_POS = calculate_mean_curve(all_mean_32C_POS_aux, max_values[2][3])

    plot_final_data_healthy(all_mean_1A_PRE, all_std_dev_1A_PRE, all_mean_1A_POS, all_std_dev_1A_POS, all_mean_1B_PRE, all_std_dev_1B_PRE, all_mean_1B_POS, all_std_dev_1B_POS, all_mean_1C_PRE, all_std_dev_1C_PRE, all_mean_1C_DURING, all_std_dev_1C_DURING, all_mean_1C_POS, all_std_dev_1C_POS, -1, 1, True)

    plot_final_data_healthy(all_mean_2A_PRE, all_std_dev_2A_PRE, all_mean_2A_POS, all_std_dev_2A_POS, all_mean_2B_PRE, all_std_dev_2B_PRE, all_mean_2B_POS, all_std_dev_2B_POS, all_mean_2C_PRE, all_std_dev_2C_PRE, all_mean_2C_DURING, all_std_dev_2C_DURING, all_mean_2C_POS, all_std_dev_2C_POS, -1, 2, True)

    plot_final_data_healthy(all_mean_31A_PRE, all_std_dev_31A_PRE, all_mean_31A_POS, all_std_dev_31A_POS, all_mean_31B_PRE, all_std_dev_31B_PRE, all_mean_31B_POS, all_std_dev_31B_POS, all_mean_31C_PRE, all_std_dev_31C_PRE, all_mean_31C_DURING, all_std_dev_31C_DURING, all_mean_31C_POS, all_std_dev_31C_POS, -1, 31, True)

    plot_final_data_healthy(all_mean_32A_PRE, all_std_dev_32A_PRE, all_mean_32A_POS, all_std_dev_32A_POS, all_mean_32B_PRE, all_std_dev_32B_PRE, all_mean_32B_POS, all_std_dev_32B_POS, all_mean_32C_PRE, all_std_dev_32C_PRE, all_mean_32C_DURING, all_std_dev_32C_DURING, all_mean_32C_POS, all_std_dev_32C_POS, -1, 32, True)

    return all_mean_1A_PRE, all_mean_1A_POS, all_mean_2A_PRE, all_mean_2A_POS, all_mean_31A_PRE, all_mean_31A_POS, all_mean_32A_PRE, all_mean_32A_POS, all_mean_1B_PRE, all_mean_1B_POS, all_mean_2B_PRE, all_mean_2B_POS, all_mean_31B_PRE, all_mean_31B_POS, all_mean_32B_PRE, all_mean_32B_POS, all_mean_1C_PRE, all_mean_1C_DURING, all_mean_1C_POS, all_mean_2C_PRE, all_mean_2C_DURING, all_mean_2C_POS, all_mean_31C_PRE, all_mean_31C_DURING, all_mean_31C_POS, all_mean_32C_PRE, all_mean_32C_DURING, all_mean_32C_POS, all_std_dev_1A_PRE, all_std_dev_1A_POS, all_std_dev_2A_PRE, all_std_dev_2A_POS, all_std_dev_31A_PRE, all_std_dev_31A_POS, all_std_dev_32A_PRE, all_std_dev_32A_POS, all_std_dev_1B_PRE, all_std_dev_1B_POS, all_std_dev_2B_PRE, all_std_dev_2B_POS, all_std_dev_31B_PRE, all_std_dev_31B_POS, all_std_dev_32B_PRE, all_std_dev_32B_POS, all_std_dev_1C_PRE, all_std_dev_1C_DURING, all_std_dev_1C_POS, all_std_dev_2C_PRE, all_std_dev_2C_DURING, all_std_dev_2C_POS, all_std_dev_31C_PRE, all_std_dev_31C_DURING, all_std_dev_31C_POS, all_std_dev_32C_PRE, all_std_dev_32C_DURING, all_std_dev_32C_POS
          
if __name__ == '__main__':

    number_healthy = 8
    number_PD = 1
    healthy_height = [1.538, 1.74, 1.8, 1.79, 1.63, 1.72, 1.72, 1.604]
    session_order = ['B-C-A', 'B-C-A', 'C-B-A', 'C-A-B', 'A-B-C', 'A-B-C', 'B-A-C', 'A-C-B']
    PD_height = [1.7]
    PD_activity = [1] # The only PD patient performed ACT1
    activity1_headers = ['Position Right Foot z', 'Position Left Foot z', 'Velocity Right Foot z', 'Velocity Left Foot z']
    activity2_headers = ['Position COM x','Position COM y']
    activity32_headers = ['Position Right Hand z', 'Position Left Hand z', 'Velocity Right Hand y', 'Velocity Left Hand y']
    activity31_headers = ['Rest Right Hand x', 'Rest Right Hand y', 'Rest Right Hand z', 'Rest Left Hand x', 'Rest Left Hand y', 'Rest Left Hand z']

    # --------------------------- REFERENCE CURVES ---------------------

    ACT1_RIGHT_FOOT_POS = [
        0.08292806322973532,0.08357521050924398,0.0842351553790875,0.08498682482644082,0.08576313169860371,0.08660458883965928,0.08750516865420384,0.08853594037271885,0.08970165251355328,0.09106633555658884,0.0926639105026893,0.09456644402702948,0.09674885568954489,0.09928499742674146,0.10215166878389133,0.10541438778750084,0.10916301623537644,0.1133832243732004,0.11814839723902092,0.12342155069299603,0.12921431583066348,0.13552803166022645,0.14233162438602867,0.14959400370760173,0.15729821374905784,0.16535434011965414,0.1736681059118579,0.18215574256512812,0.1907235123660637,0.1992699485403214,0.20766710064490648,0.2158030314798074,0.22364761694032928,0.23103218194484912,0.23795472089911393,0.24426559733810127,0.2499413260521539,0.254870102575345,0.2589996892062319,0.262332351966935,0.26478783302440295,0.26631580937535854,0.26690947788677555,0.2666258947666068,0.2653908546549609,0.26324595538490164,0.26026350591812697,0.25643099231779415,0.2518727427382456,0.2466106915903092,0.24073094816699706,0.2343034133592924,0.2274027612200339,0.22010976788517625,0.21249027468560167,0.20464556385254448,0.19665665861255535,0.1886346363110117,0.18064402764528342,0.17272532951652556,0.16496457564367414,0.15743827664308024,0.15017443774017245,0.14322450992339233,0.1366174019142247,0.13037321512894248,0.1245113938987581,0.11902822221896366,0.11394063124447877,0.10928156391620363,0.10502759143828404,0.10117572797201113,0.09767082778924017,0.09454560766797421,0.09173955727198907,0.0893331375894974,0.0872533926476558,0.0856083332870663,0.08435495861906046,0.0836602124484307,0.08324885106359366
    ]

    ACT1_LEFT_FOOT_POS = [
        0.08388707133709745,0.08455499990660958,0.08526504973596562,0.08610827441922332,0.08702239991702888,0.08807575571241964,0.08927996643746007,0.09068036266759133,0.09228382513066785,0.09409603813250451,0.09616545251677679,0.09851636587934715,0.10121354239566815,0.1042273430933711,0.10763892423633849,0.11146119613950121,0.11575284842718853,0.12053193530309055,0.12582133670588244,0.13161358678456375,0.13787122377177055,0.14464482817646795,0.15186072852705254,0.15949033585435116,0.16746232667832575,0.17571051321657746,0.18412638186135077,0.19263297632272977,0.2011515864312148,0.20955685910725105,0.21774067965387245,0.22557968350352656,0.23295368578931833,0.23980936207858872,0.246086334870901,0.25170834271729847,0.2565113478368076,0.2605035700298567,0.2636021429593797,0.26578953460214644,0.267129667394234,0.2674749261967249,0.2669716989526728,0.265578698715651,0.2633097549608498,0.26022205062594755,0.25635707546170483,0.2517728005861557,0.24655763667826783,0.24071286836906522,0.23439102603929787,0.22761647662016407,0.22046641798835345,0.21303421724909008,0.20539559815608252,0.19760584324187355,0.1897526785242852,0.18190406157187203,0.17418111248275805,0.166585373293805,0.15921053038622796,0.1521069974737803,0.14528059468913854,0.13876804971512227,0.13259442534867158,0.1267867377791643,0.12136413002314698,0.11632884767000094,0.11169099680907561,0.10746853667628503,0.1036386392754179,0.10019555030904642,0.09706321237716246,0.09426975817989451,0.09178327152368648,0.0896513882130766,0.08783783108249192,0.0863887765227136,0.0852731759815444,0.08464340250551254,0.0842490609825316
    ]

    ACT1_RIGHT_FOOT_VEL = [
        0.04569084306395272,0.05395057527619194,0.06129310280066883,0.06596181713183356,0.07081140483111677,0.07658599075340737,0.0841304167310496,0.09441848495599199,0.10827081365987029,0.12564661384636447,0.14742095719299045,0.17355867940136893,0.20266040083589498,0.23429787710212657,0.2694380852568958,0.3088212095489791,0.35244759967486433,0.39955138712523713,0.44901636307368253,0.49925258207645135,0.5492707287530797,0.5979589677657612,0.6444875481472118,0.686827918011144,0.7243026133049105,0.7542848157348537,0.7768547790162065,0.7903450987090169,0.7937842939790243,0.7876563968302192,0.7706214728347057,0.7437749285940501,0.7070529364281958,0.6614386421265442,0.6080037746575672,0.547527273496177,0.48096536215115604,0.4081051912181049,0.32959514601782836,0.2479395529127074,0.16434483039578923,0.07999690112718084,-0.004656388716374933,-0.0889265621423404,-0.1717487716270857,-0.25160119213492443,-0.3276877397369862,-0.39852375694895387,-0.4637857633463618,-0.5230984989329477,-0.5763819732988152,-0.6234598842752307,-0.6633082507543008,-0.6958166919489331,-0.7204112082298644,-0.7373705192597245,-0.7472118635667179,-0.749415847125022,-0.7442118564131283,-0.7316418916175715,-0.7120435971681899,-0.6862772403738263,-0.6557375203169539,-0.6213161355590998,-0.5838498015160066,-0.545390197675813,-0.505825418112081,-0.46694014436473125,-0.4283547438824326,-0.3859721973349335,-0.3469786339432482,-0.3102580212799823,-0.2746374669236763,-0.23879721496786607,-0.20100668336892635,-0.1624504098695366,-0.12217764561748624,-0.08128266092816534,-0.04728169104920114,-0.014069796533879382,0.01784971078753238
    ]

    ACT1_LEFT_FOOT_VEL = [
        0.05287400433990309,0.06844145006694884,0.0816262223163217,0.09236727653051309,0.10380192880398091,0.11376018041451315,0.12498511661192062,0.13784621270417347,0.15225082590249725,0.1690903301026846,0.1895425394895777,0.21325010717667983,0.24055507363301418,0.27297578134996786,0.3091545833526556,0.34884140009492787,0.3922349492515826,0.4386746816096596,0.48820371596630313,0.5377019248672085,0.5870412093148195,0.6335144766737467,0.6760389972149025,0.7135000952019633,0.7444894310822571,0.7681661043897995,0.783424399541988,0.7893423107697006,0.7850369097836934,0.77087684805303,0.7469020852429059,0.712164651797024,0.6672973319912702,0.6115100071613949,0.5479378775954671,0.4765302416220772,0.40011032881224823,0.3200901427319714,0.2373119201198789,0.15348741805503788,0.06828822924514087,-0.01653064585903176,-0.10065568333295226,-0.18173520741110682,-0.2590049743087782,-0.3316804255664186,-0.3990321691497672,-0.4609169528483032,-0.5170749507747564,-0.5670946219520371,-0.6108769748318988,-0.6476992851629112,-0.6771833018292724,-0.6991168816425098,-0.7136430920953835,-0.7210153011453939,-0.7222468651739963,-0.7176084615367216,-0.7076881598402642,-0.6923031686842004,-0.6718550616834947,-0.6463517969311993,-0.6170714574822905,-0.584329330778055,-0.5495618898440567,-0.5126889988489708,-0.4745821362988558,-0.43969162924768296,-0.4006859386613891,-0.3610464484976292,-0.3237242315845702,-0.28594599185414904,-0.2501510499007307,-0.21681619254582418,-0.1824375835327061,-0.1475294873741855,-0.11438355579626501,-0.07797833577397459,-0.04077759939723728,-0.00865249707173365,0.019892306164417565
    ]

    ACT2_COM_X = [
        0.0,1.0722806604801516e-05,3.0354090854556315e-05,5.9817270617740226e-05,0.00010416914867891758,0.00015920824166676395,0.00022088513230339446,0.0002980915230061387,0.0003968477787191782,0.0005149428850221849,0.0006456272631315263,0.0007832667279819017,0.0009337184215850169,0.0010950721247004191,0.001262607448011821,0.0014410508200072836,0.0016318484591514026,0.0018336717140954029,0.002047156134660888,0.0022723848810568548,0.0025044996636804095,0.0027447240094385927,0.0029974803288371904,0.003261483272110774,0.0035407511868496694,0.0038288949739621325,0.004134269461402523,0.004443079587361037,0.004763598352202442,0.005098881886073631,0.005458362026077995,0.0058333715157525,0.006216445629076833,0.006602366226067344,0.006993429290253553,0.007398378062157637,0.007824162040137944,0.00826750182002177,0.008718177243823094,0.009170932701083684,0.009614682070914302,0.010054287848964144,0.01051343591730737,0.01097009456639915,0.011430289738250679,0.011898772586952197,0.012369335050125177,0.012838898978999674,0.013313272526040684,0.013786729311905707,0.0142657122621999,0.014746475026825537,0.015216322468472612,0.01567914120753565,0.01613481068986554,0.016582578372998022,0.017022064375998416,0.01746392926792907,0.017908930697064382,0.018359907835028763,0.018815092460594637,0.019266438340048,0.019717130628171306,0.020168349953098325,0.020613128835618028,0.021049266533685232,0.021474941384263896,0.021889865091746633,0.022295091960627414,0.022699616469834066,0.02309662328918345,0.023483238257998668,0.023860779124053724,0.024233043441762997,0.024596572115610407,0.02494366400623574,0.02526972338792302,0.025581074416925573,0.025879628587641784,0.02617006842717093,0.026450752722756182,0.02671355550678005,0.026955775190420014,0.02718040274515586,0.027392848606201683,0.027596411394539014,0.027796145271773293,0.027984053668231562,0.02816095255509908,0.02833016944174567,0.028487951544595144,0.02863137930890232,0.0287637440364107,0.028882843883934445,0.028995045580907333,0.029099862931204395,0.02919477467320196,0.02927747374487892,0.029353375279632152,0.02941992422537873,0.029475887538932904,0.02953974411331721,0.029607092775436522,0.029668872505334815,0.029725674188251103,0.02977828367232095,0.029828695317096005,0.029876390173679518,0.02992784568331182,0.02997827136405065,0.030023728660285173,0.030063834197894948,0.030102239595032512,0.03013629831992444,0.030164400208370593,0.030188707812830756,0.03020937468312744,0.03022112964306288,0.030223345706168983,0.03021934518571888,0.030212436071295964,0.030202552415039788,0.030189841755541375,0.03017564413381199,0.030155264637727094,0.03013202311756183,0.03010292670434775,0.0300703861651668,0.0300309332325666,0.029978972138373167,0.029919715729021204,0.029855345454263202,0.02978267139658672,0.02969912958859218,0.029615822867928146,0.029536670415188343,0.029449368228151016,0.029349608669751212,0.029254227046127998,0.0291498760398496,0.029038401307002276,0.02892068570052285,0.02878961192041332,0.028642100870062318,0.02848839115470296,0.028331031813170246,0.02816438513436499,0.02799450455074405,0.027818697079343216,0.027629756616349907,0.0274327614354566,0.02723524955639481,0.0270354010802403,0.026830099593185677,0.026611642106285985,0.026392397614229998,0.026175811881524583,0.025947042857476912,0.025705255222603238,0.025428807593902855,0.025137776080520882,0.024853947798394074,0.0245683747450806,0.024264982183835784,0.023951124625680692,0.023609370132521665,0.023250714960492205,0.022885897713160505,0.022529424742268947,0.022191023261954296,0.021866188090997232,0.021547853722143254,0.02121973710662535,0.02087836964725534,0.02052652802861396,0.02016162929767859,0.01978395140930851,0.01941378142772507,0.019033226922639562,0.018630067192124775,0.018217560931931526,0.01780332570618863,0.017372533821156694,0.016930519968459026,0.01648431607832343,0.01603643579290423,0.015586498232969731,0.01513410878782498,0.01467216236852767,0.01421174553703805,0.013754423915469396,0.013293940275680122,0.012841274990887316,0.012405552743115139,0.011987294669512275,0.011584990449825291,0.01119928379364346,0.010823959855441143,0.010461726455526163,0.010104173965580318,0.009743623068309976,0.009379426741579128,0.009015344270082257,0.008651360197375862,0.008296983005061861,0.007952039575390438,0.00761086551083659,0.007273259837599841,0.006938052732604533,0.006606274044897234,0.006281540883883316,0.00596585160872358,0.005656946782866032,0.005349943485739264,0.0050476490157356085,0.004761321503905749,0.004482706014648022,0.004202965189797375,0.003920898796832481,0.003636366189445378,0.0033502232302856233,0.0030686261449086316,0.0027939484280378873,0.0025356077555582224,0.0022897711196124826,0.0020507277057987827,0.001820151150170063,0.001597984574314315,0.001383070125589081,0.0011720789566770917,0.0009618603491235722,0.0007629088714893872,0.0005758152623427937,0.0003946906795254945,0.00022380402538763056,6.591060829957973e-05,-7.504312270104063e-05,-0.00020452254463604325,-0.000322997001613895,-0.0004272784743985514,-0.0005202842790105744,-0.0005983555985250577,-0.0006636236775885707,-0.000722777527052243,-0.0007733814532654981,-0.0008151349742557268,-0.0008515921058335485,-0.000887898681469157,-0.0009109042986001903,-0.0009242784376188862,-0.0009297213802766935
    ]

    ACT2_COM_Y = [
        -0.1849739492612176,-0.18479354360932598,-0.18456400307281262,-0.18427619940877069,-0.18392203742588684,-0.1835073776200986,-0.18303361244137895,-0.18248173665420606,-0.18186513341511704,-0.18117539686774475,-0.18041506472685243,-0.17958361808653364,-0.1786783799802222,-0.17770466526232323,-0.17666048195715098,-0.17554874183159433,-0.17436857180189766,-0.17311239564941072,-0.1717827933942215,-0.17038008372210686,-0.16889580690970943,-0.16733136867469792,-0.16568671753905376,-0.16396545634725154,-0.16215884642429604,-0.16026977900564252,-0.1582978003996343,-0.15623862917801482,-0.1540962368000842,-0.15187059779718456,-0.14955997442904628,-0.14717019816335225,-0.1446976318573531,-0.1421560410696131,-0.13953921456757631,-0.13685069800082975,-0.13409868529219277,-0.1312815616774369,-0.12842630925852053,-0.12552188999732655,-0.1225690837274586,-0.11956611867148541,-0.1165176506524816,-0.11341721464799726,-0.11028746597821909,-0.10713203447834503,-0.10394797145159666,-0.10075914239497913,-0.09757957519643669,-0.09441567691894183,-0.09127224769107728,-0.08814894236901993,-0.08505295439414824,-0.08198830663414267,-0.0789671178513794,-0.0759952439418901,-0.07307689769619344,-0.07021101556311594,-0.0674038866270917,-0.06465618290345304,-0.06195988405105487,-0.059310960050502425,-0.05672091425811984,-0.05418306801263146,-0.05170125033907679,-0.049294277949680505,-0.046960057195755696,-0.04468733380868917,-0.04244545632569513,-0.040247904856195574,-0.03810149409283715,-0.0360080905014439,-0.0339682299020942,-0.03197110176329849,-0.030009143594379065,-0.028092144805592228,-0.02621687078486097,-0.02437800592355864,-0.02256039727874794,-0.020772757036890018,-0.01902686806590108,-0.017332060226568285,-0.01569056564090178,-0.01407829257521312,-0.012486786372098825,-0.010910877589172405,-0.009356803238359046,-0.007837449865993806,-0.006354213680215645,-0.004913110845340715,-0.0035076209005110554,-0.0021506951024516407,-0.0008309105349253226,0.0004607060406306214,0.001720715073166472,0.00293889940905559,0.004112201083788882,0.005245141023942649,0.0063176015634843485,0.007356715261906267,0.008361484502352759,0.009319047631115723,0.010217053429195494,0.01107002130829298,0.011875833997493493,0.012635722321384072,0.013336484162034192,0.013995958232665607,0.014581899856371587,0.015110091017979941,0.015574660072222779,0.01598289800664604,0.016328090838075706,0.01662269491022143,0.01686253472858971,0.01704775594126238,0.017183768268572715,0.017280752001997027,0.017332447331629385,0.017341586011489853,0.017296373582124467,0.017200941637897257,0.01705672383950668,0.016870970801492434,0.016638393543015526,0.01636727026457675,0.016046649338593965,0.015681913222528446,0.015265779928880821,0.014802157164631955,0.014292924564370419,0.01374245865813638,0.013147566000282768,0.012514832302147153,0.01184764318710952,0.011145210455422896,0.010393938479914058,0.009600332865713176,0.008753278853415694,0.007863257829363552,0.006923929981236009,0.005948012852257119,0.004937573678037674,0.0038992671746452914,0.002819081583422735,0.001700540671341423,0.0005317048084720349,-0.0006734048774142652,-0.0019132089151444414,-0.003185508092490293,-0.004482028962949427,-0.0057990874783280565,-0.007156510619941855,-0.008552814536229953,-0.009983942196190178,-0.011428757546693097,-0.012883260249333825,-0.01435601969934952,-0.01585715963653173,-0.017397900026376516,-0.01896719851169008,-0.020551783974937832,-0.022156085614563937,-0.023789759935482397,-0.025449701258793717,-0.0271470982342633,-0.028897941647914576,-0.03070136440449381,-0.032545265352031595,-0.034415618903833496,-0.036308281088660264,-0.03823698247787235,-0.04021669991701672,-0.042237734220081403,-0.04430164903817826,-0.04641076095101395,-0.04855735250465296,-0.05073603201213506,-0.05295260896543936,-0.055224876957397785,-0.0575540560230006,-0.05992637950623497,-0.06234640550837418,-0.06482257687177571,-0.0673463142525814,-0.06991118206439782,-0.07251512407507242,-0.07515939573586275,-0.07785003137900583,-0.08057084732160608,-0.08331686847827274,-0.0860819484693302,-0.08886414780508899,-0.09165838040366346,-0.0944466863255471,-0.0972221356897121,-0.0999729528991361,-0.10268707244456705,-0.10536127373469177,-0.10800376484398931,-0.11062271249622439,-0.11322307957510495,-0.11579613744455586,-0.11834066438750539,-0.1208585271028196,-0.12334401136029931,-0.12579870573418597,-0.12822105685367757,-0.13060722963460272,-0.13295159187723535,-0.13524909773884597,-0.13749967067524346,-0.1397102250495222,-0.14187993813371955,-0.1440080808604615,-0.14608814601406953,-0.14813238017087385,-0.15013650333464376,-0.15210011191157455,-0.1540160607718625,-0.15588528046222913,-0.15770886028910877,-0.1594855526250323,-0.1612101422167355,-0.16287943975187694,-0.16448504277283177,-0.16602053396003705,-0.16748932476797915,-0.16888685085276842,-0.17022963352375242,-0.17150910309100945,-0.17272140681642434,-0.17386264733305734,-0.17494498702968872,-0.17596281124268776,-0.17692073947797338,-0.17781042229425167,-0.17863950716026375,-0.17940810543378535,-0.18011595309941472,-0.18076468237987867,-0.18135598057507865,-0.18188982744814863,-0.1823606657453092,-0.18277724647794238,-0.1831285184678579,-0.18342542884453694,-0.18365514969785365,-0.18381798018824216,-0.18391364142808586,-0.18395437099098397
    ]

    ACT31_RIGHT_HAND_X = [
        0.0,-7.704920039905962e-07,-8.816851054625752e-06,-2.082232730438229e-05,-2.8662034156625724e-05,-2.6630487908947657e-05,-4.367106928797457e-05,8.576828249864923e-05,0.00017556250320338766,0.00020532934490651215,0.00019218294603920967,0.00013001094428246232,8.331355745350287e-05,3.451203478059453e-05,-4.822575606020595e-05,-0.0001159350610251019,-0.00016411598875434158,-0.0001908247259950717,-0.00016740085104896421,-0.0001409352631878078,-0.0001290885656212174,-0.00010760014871038813,-9.437110913518786e-05,-6.827386555438194e-05,-8.392324477600977e-06,3.1532657811083235e-05,3.093400713555529e-05,3.15821570193605e-05,3.269366247471526e-05,1.1987957434225183e-05,8.403450564974294e-06,-2.197610253302609e-06,-1.7219259918303855e-05,-1.4541005822845004e-05,8.010716549903093e-06,-1.6930654341774868e-05,-2.3126780283710903e-05,-1.2372637785368263e-07,3.0003517753105336e-05,6.739866744628614e-05,6.108433954591813e-05,2.4095047935266642e-05,-1.0738420590552394e-05,-2.5026335477405483e-05,-3.935240642410567e-05,-2.805220103257495e-05,-2.5366846029844723e-05,-4.4835103545548146e-05,-5.1948574413829766e-05,-2.5150813972447592e-05,5.190091471634924e-06,3.6798706429810464e-05,4.864784657165886e-05,3.930360951627713e-05,3.005538581047941e-05,4.0927026597243074e-05,4.939542380015733e-05,4.916758765648849e-05,4.527351142203262e-05,4.052974008114515e-05,4.025797641539635e-05,4.752067245607184e-05,4.5920115479055125e-05,4.9278380983490325e-05,4.7835582119824436e-05,4.2932594545155575e-05,3.949721812081581e-05,4.501552192636636e-05,3.181494536228796e-05,5.547451310512893e-06,-2.3570991998536978e-05,-4.071172610439329e-05,-5.6533054155175884e-05,-5.8193354381179685e-05,-6.517407096543177e-05,-9.657980316210442e-05,-0.00010391807364518848,-9.371665008726209e-05,-0.00010091258253163312,-8.281091037732306e-05,-6.224241350926245e-05,-3.210196842276414e-05,-7.05237184125818e-06,2.147501849330301e-05,3.471982305192896e-05,5.3313356971095244e-05,6.371443643765672e-05,7.269683839257957e-05,8.711679388028728e-05,9.728640066182109e-05,0.00010624768502425341,8.58818269722589e-05,2.036641132860468e-05,6.042527854186487e-07,1.262108618905063e-06,-1.2440052885688256e-05,-1.8415293919180192e-05,-1.0423118366667684e-05,8.306403812107809e-06,2.3403201583553956e-05,1.770681785597e-05,1.1954897716947677e-05,3.911690446136974e-06,-1.310309842513625e-05,-3.32015620739225e-05,-4.528677831805338e-05,-4.6852504050501084e-05,-3.9205056569159847e-05,-2.3268477381127722e-05,-4.0112641373056266e-05,-3.872596653647731e-05,-3.6290886324139136e-05,-3.3813055157077524e-05,-2.74590773339005e-05,-1.124686368077566e-05,-6.892276184119516e-06,-3.096853099349284e-06,9.687470417216617e-06,2.7059845467903854e-05,4.099124445729177e-05,4.6621821193531984e-05,5.22264858813592e-05,4.530140020417788e-05,4.250067798703699e-05,2.7755185333536815e-05,1.1501433003570095e-06,-2.0975548794529702e-05,-4.2356714777574415e-05,-5.302215374788601e-05,-6.595998745493666e-05,-7.623453524946339e-05,-7.619615834940687e-05,-6.473255853768063e-05,-4.7078986433997953e-05,-2.403774468843585e-05,-6.5830855607277375e-06,7.46793634424685e-06,1.376088791192066e-05,2.8277838521359705e-05,4.6075744154527084e-05,6.333926699155443e-05,7.037343222054919e-05,1.6014611338081686e-05,-6.704243361539652e-06,-2.659837240215552e-05,-4.5057952794164075e-05,-5.8313356361214316e-05,-7.503498956000441e-05,-7.33627193206725e-05,-5.827689963641265e-05,-1.6211320646618487e-05,3.548539837522404e-05,0.0001009009687227508,0.00011964605711731446,0.00012578878301928144,0.00012628039702490424,0.00012823804553712718,0.00011113998527289964,0.0001003348785195286,7.776974942851562e-05,3.9797895199072584e-05,1.2779557771400897e-05,-3.454763591657396e-05,-5.1824694481129136e-05,-6.98561256987582e-05,-8.735955659564654e-05,-0.00010538346872099567,-0.00012039173404640188,-0.00010465560939252893,-6.22140775821886e-05,-2.511337917509698e-05,-2.4923258120577113e-05,-2.1356401826839804e-05,1.6681070366964327e-06,3.695741103399665e-05,6.256027465520472e-05,0.0003632412043355138,0.0005484780250889258,0.00043042037812363753,0.00022229417383299015,4.503456072379148e-05,-0.0002544166225670733,-0.000689680942393178,-0.0008815423385954182,-0.0008168767219046124,-0.0005776728987365915,-2.9948785423057215e-05,0.0010736527962186867,0.001547875322130632,0.0014785019128905341,0.0010725366253587147,0.00025656486935571294,-0.0019192974518916852,-0.0020286049902812587,-0.00172120820392667,-0.0012374226429789496,-0.0006906552573915344,-0.0003300634886807407,0.0005977511888441159,0.001136977681224529,0.001354199578302201,0.0013124934441426288,0.001449592850609257,0.000780983734188051,0.00016873972879083017,-0.00022651226848943845,-0.0005174458713043192,-0.00042341474204424636,-0.00038724835680348804,-0.00028723028753876604,-0.00023609259095257502,-0.00019206795954314015,-0.0001311716324642624,-3.9305497160568624e-05,3.5886223416934585e-05,5.058757245301345e-05,3.30309545810319e-05,-3.3065657503772405e-05,-6.346980270167841e-05,-3.400004007820742e-05,-2.3737534787738294e-05,-3.0908656771821704e-05,6.47323298894367e-05,0.00023085437596610833,0.00016823180419676996,7.925327560802812e-05,-2.695188945901689e-05,-0.00012389795688488186,-0.00019936711458188303,-0.00020875407722179302,-0.0001846626028667118,-0.00014579863302615172,-0.0001166469473580031,-0.00018859220079732114,-0.00011631631548811464,-3.0013651285018387e-05,5.927042083892381e-05,0.00011308736399886273,0.00031665110391668286,0.0002894259486993531,0.0002278815270794162,0.00014736736626979883,6.033438170329201e-05,0.00010854883330020274,9.011706377706393e-06,-8.144594263800196e-05,-0.00013454911061124702,-0.00014339615039638367,-0.0002765018408312148,-0.00028591693117075405,-0.00019984345758307105,-9.170176415320775e-05,2.8272582457043287e-06,7.268240928530569e-05,0.00012206234395183676,0.00014473272013474056,0.0001574261685193364,0.00015043639316559914,8.100453145238083e-05,-1.7832320895189793e-05,-3.698475236468398e-05,-2.9062001452559702e-05,-6.41886788353428e-06,8.732104017209593e-05,0.00028032070850347185,0.00024062647164717942,0.0001666648780275161,8.449849738550854e-05,2.079648892725129e-05,4.204483626444273e-05,-4.521346329540714e-05,-0.00012151334696091992,-0.00016746763853787947,-0.00018633017192740248,-0.0001878138305529812,-0.00018923525211347634,-0.00016878970548967766,-0.00012499300002462478,-7.223461528009602e-05,7.033868422371432e-05,8.024229907423786e-05,6.890060669107327e-05,5.782675776171594e-05,4.4303993696971136e-05,1.7376457655333323e-05,7.967223701786595e-06,5.5384721437189606e-06,1.793090598985935e-05,4.2007868746909005e-05,1.3350466707773713e-05,-3.1797875148965347e-06,1.2395940362989238e-05,3.0455800966265734e-05,4.0648547141085424e-05,5.646374238972379e-05,6.0728044152827836e-05,3.856913225224377e-05,1.1857628407025303e-05,-2.1604527356077977e-05,-2.4429482374102423e-05,-2.0934512918521556e-05,-5.082046878183266e-05,-8.002724688109445e-05,-0.00010604377882367161,-0.00010066417877600506,-7.315628868074462e-05,-7.18130490523217e-05,-7.314535869884416e-05,-6.186542944422027e-05,-3.483235190383095e-05,-1.493555395636869e-05,1.8442078136124555e-05,3.927455685980691e-05,5.8674622343613676e-05,6.407968905109333e-05,0.0001054808301284965,8.392143273778445e-05,4.565741609639244e-05,7.010308677307388e-06,-2.7432460762796326e-05,-6.298404577229304e-05,-7.781284472557759e-05,-5.874947070104514e-05,-4.5160579825523405e-05,-2.9301301753870426e-05,-1.6569209395939417e-05,-5.435903255699946e-06,1.1461077063387094e-05,4.592650898805265e-05,6.225578823904129e-05,7.408476933199364e-05,7.92211990620016e-05,7.166899466280728e-05,5.834410647666375e-05,4.4144614837111565e-05,3.136517623361825e-05,3.071787732140017e-05,2.9321567752792123e-05,2.2510044393424934e-05,1.5809338168221565e-05,-9.92021529926207e-07,-4.137633386389081e-05,-5.1754507957048195e-05,-5.5158707206096286e-05,-5.5202777769998875e-05,-3.391919596911056e-05,2.2473786606774606e-05,2.950597593674552e-05,2.6119785888187885e-05,2.5294561344060175e-05,2.403108061919233e-05,-3.2615435406871347e-06,-2.1244163532886494e-05,-3.0381720142376547e-05,-3.363616613674984e-05,-4.856739386728626e-05,-9.284332013393997e-05,-0.00010206610139868957,-0.00010496712303506135,-9.484458529455488e-05,-7.137063628382912e-05,-5.310061372021928e-05,-3.4995189232209382e-06,3.633407132889335e-05,5.5425619015376974e-05,7.43575834468369e-05,7.057918991368796e-05,8.45286493956252e-05,0.00011835986128074035,0.00011551655171184393,9.398253291681256e-05,6.526552191483838e-05,4.249399466868212e-05,2.8386649485596312e-05,1.9548176784404654e-05,3.1624453839697604e-06,-1.0782521695012337e-05,1.3246810644307484e-06,-8.43711262681454e-06,-2.5466653253742793e-05,-4.236167908279985e-05,-5.7695779012215875e-05,-0.00010367198250359005,-6.275575481301745e-05,-2.7371080430196838e-05,-3.9661780082971486e-06,1.852898560545506e-05,9.720932445390018e-06,4.388647225902151e-05,5.5142922460914134e-05,6.142012692710533e-05,5.045543240129947e-05,4.1551985476904266e-05,2.3518463898874857e-05,7.449623528099227e-07,-2.6011161980360198e-05,-5.613667014612954e-05,-0.0001087886837841927,-0.000142100874646223,-0.00014685414987865032,-0.0001269347164853067,-8.980134074369039e-05,-4.07006742445365e-05,-4.4826335504804176e-07,1.9198288295829577e-05,3.3778188362059715e-05,5.1380882570278015e-05,7.830112424107436e-05,0.00010468593773171532,9.259907770058942e-05,6.844003414963634e-05,5.019085937238611e-05,3.736048577291221e-05,3.064699331953615e-05,3.0746772533751515e-05,2.846926123130502e-05,2.7480438083098444e-05,2.8027636780442406e-05,5.150461742937262e-05,5.02072723935649e-05,4.277051689538652e-05,1.8489486213852754e-05,-2.01395715173198e-05,-5.646798381761446e-06,-5.815528871255493e-05,-9.905263320279117e-05,-0.00011303355415185857,-0.00011912833324622838,-0.00010035656412762257,-8.863823619679003e-05,-6.882776565913326e-05,-3.2597640978067694e-05,1.231279497482463e-06,6.193480493550572e-05,0.00010609886889954442,0.00013008310954448237,0.00013856897973260144,0.00013680247567331028,0.0001405008023873139,0.00012056728272189426,8.391604514646675e-05,4.349892919344052e-05,-4.599195449971288e-06,-4.9203581882894587e-05,-7.381506574237144e-05,-8.615309149681907e-05,-9.522465606787369e-05,-9.165348515787561e-05,-6.0789281795107706e-05,-3.402954883117263e-05,-3.2393872955098e-05,-4.1408086905114914e-05,-5.55982025584285e-05,-6.386885792368902e-05,-4.906034099937121e-05,-5.832992856672965e-05,-6.612439744376875e-05,-7.301782401777754e-05,-6.188733017632356e-05,1.7177393844005516e-05,2.584694101320918e-05,3.785562746847277e-05,4.597378665103734e-05,4.05764656674134e-05,5.06216539948488e-05,5.5972769681992015e-05,4.6920361652663374e-05,3.1898470665153755e-05,9.725352061244923e-06,3.720704674778353e-05,4.074357285222102e-05,4.673956523623473e-05,4.855054171785008e-05,5.222139426507813e-05,2.4898624918415864e-05,2.259583956506772e-05,3.9139425344433177e-05,4.203407399756032e-05,3.630637993927303e-05,6.223201752946486e-05,6.575814580873507e-05,3.263982712375465e-05,2.823210787818022e-06,-2.696961677099192e-05,-5.1943174787207434e-05,-6.90591629274296e-05,-8.185517002337583e-05,-8.588312476219578e-05,-7.408857574295964e-05,-7.313787619791723e-05,-9.776422952826355e-05,-8.931014260633054e-05,-6.916525669707686e-05,-3.2497107650164354e-05,1.9139199160651876e-05,9.709309532470046e-05,9.496360191463293e-05,7.700881307750995e-05,5.7934793979317404e-05,3.6792991406890674e-05,3.745330075274507e-05,1.2942814394513895e-05,-1.6796758669544796e-05,-4.090552365638347e-05,-5.427998864008225e-05,-2.6434129478658828e-05,-2.244176489942707e-05,-2.7454750682658303e-05,-3.09132019478984e-05,-2.743109799242173e-05,3.495050556231985e-05,4.920673505946907e-05,2.6381789066864248e-05,-6.922434106789753e-08,-3.2697794621605964e-05,-7.1274598892874716e-06,3.991695292242321e-06,-1.782948452317594e-05,-3.395856712159204e-05,-4.33177456831633e-05,-4.2939995440258166e-05,-4.056972989671421e-05,-3.797963017694621e-05,-3.380899445471609e-05,-1.656518543689297e-05,9.18475203287966e-06,4.1681042985833795e-05,3.878434635017081e-05,2.4504817963202062e-05,1.4199669263737004e-05,1.1613090380219929e-05,7.0303059695341155e-06,-1.1166103275124434e-07,-9.681960275894762e-06,-1.732009224079136e-05,-1.705010251552772e-05,1.790011906307093e-05,1.919618417580954e-05,4.0795092226847875e-06,-2.207070582803737e-05,-2.961879520784674e-05,-5.522551577882814e-05,-4.4224496428776784e-05,-3.0513376835631575e-05,-1.6454693312536165e-05,-4.815464342458159e-06,8.975749311573862e-06,3.874757576603968e-05,5.2971542617704836e-05,5.560578775953695e-05,5.447121460794793e-05,7.024045894036311e-05,6.821767704149973e-05,5.1390903300420554e-05,2.423577790139444e-05,4.828309539074937e-06,-2.5774959669081592e-05,-3.894078591857088e-05,-3.126397729611734e-05,-2.1641411195711197e-05,-1.5425099792348988e-05,-5.105596774704082e-06,4.369814829347409e-06,1.7540461756450183e-05,2.00583058024076e-05,1.7023252733519293e-05,8.794691646781134e-06,4.5627260363205075e-06,9.316978038219137e-06,1.0780841949237793e-05,2.4972841141347506e-06,-1.0712644377854708e-05,-4.06482096955334e-06,-4.9605497939755235e-06,-8.299422270673694e-06,-1.3029133224456259e-05,-5.797466810200149e-07,7.179940558531749e-06,1.5310080459281393e-05,1.1699633192159916e-05,1.0944652468782395e-05,8.420723476750167e-06,2.5739710778481803e-05,1.9283847320276952e-05,5.353499402022374e-07,-1.9823951704035308e-05,-3.221863518326384e-05,-4.4287811215205935e-05,-4.241345270203952e-05,-5.0654247908832555e-05,-4.797029679804376e-05,-3.013539752035932e-05,-6.432532135034175e-06,1.303983111031146e-05,3.19278345158221e-05,4.3062483083886026e-05,5.282889430913175e-05,5.1727012268739106e-05,4.757975850822914e-05,4.113114321157818e-05,3.924374069305787e-05,3.174069045842753e-05,2.1187973156514982e-05,1.0172099193488274e-05,4.765941656516411e-08,-8.373297491261378e-06,-8.421824130184624e-06,-1.1402348008481518e-05,-1.4054369638176309e-05,-8.688927760469425e-06,-4.3018038571303905e-06,-7.154430966473791e-06,-1.0345648786023645e-05,-1.359987298645064e-05,-1.5546857312305402e-05,-1.8104892246128843e-05,-2.3007948038012523e-05,-2.8101128237188322e-05,2.095710343276859e-06,1.1395679925873468e-06,-7.208525852972059e-06,-1.9825372652841583e-05,-3.507287598963756e-05,-2.76524492492468e-05,-1.6508954537898988e-05,-8.221395455865558e-06,1.0619482281322493e-05,1.766231114556833e-05,1.8490920951418324e-05,1.8790058482178577e-05,2.1929885479385683e-05,2.5456564122401104e-05,2.59015836776152e-05,3.362285271812413e-05,3.206571382713769e-05,1.776586069446671e-06,-1.0930416699851821e-05,-2.0422351075981775e-05,-3.1136352186982826e-05,-4.238256604178483e-05,-6.759351600171629e-05,-6.036742539034415e-05,-2.626668390562562e-05,1.4736389148735206e-05,3.192577130393413e-05,3.455961655725787e-05,4.6462932053442046e-05,6.510715239480738e-05,7.827452227011153e-05,8.186089940602742e-05,4.3601146106028336e-05,6.184289320937389e-06,-3.1126119764741635e-05,-6.601266304032878e-05,-6.43375381353875e-05,-7.249483800809411e-05,-8.41375417004851e-05,-9.332673693460791e-05,-9.163086724505444e-05,-7.755730523779178e-05,-4.1180414595000755e-05,-6.406731830215781e-06,2.444046915884254e-05,4.399932147843954e-05,6.824521499887495e-05,7.752241404752219e-05,8.54788925785767e-05,7.742003174745678e-05,6.564536229990287e-05,6.200695188737863e-05,5.529043925231112e-05,2.48063559099887e-05,-7.049606575076116e-06,-3.335875571930305e-05,-4.284778446847378e-05,-4.308253997924323e-05,-4.519745902515943e-05,-3.9917866308811204e-05,-2.98492864548182e-05,-2.7087965624414423e-05,-2.1632481556136193e-05,-3.369570679241495e-05,-3.5119313657817615e-05,-2.8735704707657312e-05,-3.194735093771701e-05,-3.228029182501827e-05,-2.7844810228550904e-05,-2.0975084943050788e-05,-1.3409025603067912e-05,4.7484299855086144e-06,-1.984493101529144e-06,-8.386261510250294e-06,-1.68373614641118e-05,-1.3300942803752298e-05,-1.5904170784996838e-06,1.530795652256513e-05,3.197939651292623e-05,5.3683973467180374e-05,6.633456655141968e-05,7.108303664169744e-05,7.505284992200311e-05,6.237939006419983e-05,3.186745295791063e-05,1.0696496282945898e-05,-8.879161514707593e-06,-1.6451709795249346e-05,-2.5085750398118126e-05,-3.519701043665633e-05,-3.796643115120165e-05,-4.182064082826139e-05,-5.350648130045039e-05,-4.978406434522387e-05,-3.601911360326108e-05,-2.8986564059765523e-05,-2.631120534606663e-05,-2.7396376594688736e-05,-2.690806209462372e-05,6.050822129371158e-06,4.3501889945465536e-05,4.6368115149801506e-05,3.3750636962489034e-05,3.006089552230084e-05,5.0872688785475965e-05,0.0001268125466625298,0.0002464968921995206,0.00030211289890566263,0.000250454780848144,0.00011579024103140828,-1.1863345996477616e-05,-7.891290032350984e-05,-0.00013403959167011763,-0.00021828592535676076
    ]

    ACT31_RIGHT_HAND_Y = [
        0.0,-5.185642827202954e-07,-1.2915899032231767e-05,-2.2882435240640653e-05,-2.5089851365158588e-05,-2.653874916693099e-05,-7.89464437579167e-05,-0.00027531454031893036,-0.00029337313366937796,-0.00022545940944582705,-0.0001271147962666136,-2.9215966680123604e-05,-7.940367110733893e-05,-0.00022316578819248087,-0.00014850727063045718,3.1618843148660554e-06,0.0001507549726185238,0.00022977159105883296,9.70460415021228e-05,-7.251933808368418e-05,-3.281759489908745e-05,7.725237198240736e-05,0.00017264316622330207,0.00016102862529931904,7.058945331582137e-07,-0.00012026635169680676,-6.877578316280532e-05,4.381321254606219e-05,0.00010840026748763266,0.00011720559375292253,3.279191392301707e-05,-2.1286378236280403e-05,2.853241433001357e-05,0.00010995455611894458,0.00011986422189279063,0.00011158101806845235,8.556021250866464e-05,4.0314741227141515e-05,2.7746839601473854e-05,3.237958862621684e-06,-1.6291833835770993e-05,-2.3531438724439826e-05,-3.9255188854838775e-06,1.0296807273480722e-05,3.765522427556667e-05,1.8672522999887047e-05,1.0507149618956437e-05,2.396703434162867e-05,3.8916787606181444e-05,4.0309059567056545e-05,3.938753351060481e-05,9.703489903535135e-06,-9.95646871427438e-06,-4.0254703620902053e-07,-2.4630046784757977e-06,-8.494825780361334e-06,-3.12713105164326e-07,-1.623799304273851e-05,-2.179205561570358e-05,-1.705723152001365e-06,-7.969524061295137e-06,-3.1236017006296177e-06,2.059926470171965e-05,1.167174196208879e-05,-1.4759166316721394e-05,-3.158457216031414e-05,-3.803367539564039e-05,-2.7928399904334952e-05,-4.145843144927566e-06,1.1225057681846077e-06,-6.404806495261814e-06,-6.924105961627781e-06,-2.41682685707487e-06,-1.2783967953216872e-05,-3.516943642816917e-06,-8.82014510745001e-06,-2.3815846397155317e-05,-3.166413465275137e-05,-1.678635868954921e-05,-1.8736112541107077e-05,2.4919268354493453e-06,1.6461555309823334e-06,-1.5835980154054592e-05,-2.1356181688658584e-05,1.920207876815223e-06,1.2366432854163569e-05,1.859311178650382e-05,2.0880303585434017e-06,-2.539973470751829e-05,-3.663798009980091e-05,-4.0511062937645586e-05,-4.0130465653171746e-05,-2.7305789875653234e-05,-2.2686962842451293e-05,-1.3578381300754767e-05,1.6147915011465322e-05,4.164319287019749e-05,4.8748322303140565e-05,4.094318577127859e-05,2.9851625759227034e-05,3.136820387579519e-05,3.875311945015787e-05,4.0911365182931876e-05,2.8890285816441125e-05,7.2367161514906755e-06,-6.746486043412564e-06,-1.573335620122886e-05,-1.588274761250951e-05,-1.0540514877384931e-05,1.8406702550046591e-06,-2.737497018843372e-06,6.106796289483239e-06,1.8552254429475207e-05,1.9619408960159404e-05,4.665723770101641e-06,-2.266032484432242e-05,-3.230784036224597e-05,-2.7165503395204903e-05,-1.2676998795695373e-05,9.70308840910198e-06,1.8171984918098972e-05,7.160274259221091e-06,8.834669671723811e-07,-4.9640470587253495e-06,-2.4773143778651364e-07,4.535377835252833e-06,3.3646758987436485e-06,5.166896220808716e-06,2.547159865868511e-05,4.6739664763409136e-05,5.4457867247199866e-05,4.886880258547615e-05,2.7533002149198963e-05,1.3513037847737286e-05,1.7051112162299915e-05,2.3703900303146284e-05,2.70681499110344e-05,2.462178759573454e-05,1.0731831902116497e-05,-1.5670019682653003e-05,-3.4144747057868383e-05,-3.5475196336754e-05,-2.6455689940153627e-05,-2.5583800280395208e-05,-2.2116911605181292e-05,-1.5490763031498817e-05,-9.807728337416168e-06,-4.542441947517314e-07,2.7739736512837e-06,9.839456695596755e-07,-1.870183357466506e-05,-3.451754002332953e-05,-4.684858934287718e-05,-5.1105711717327325e-05,-5.44162813070075e-05,-5.2857562364066514e-05,-5.4428886698943275e-05,-4.943209684394324e-05,-2.3158979868678612e-05,1.7550855593668617e-05,5.2191969651351334e-05,7.181216499253321e-05,8.640052208083454e-05,8.601588458293236e-05,7.263392630793058e-05,4.427051673180137e-05,2.3344454819221306e-05,8.407621527339421e-06,3.636513072193562e-06,-5.893334214264007e-06,-1.8554065796274976e-05,-2.3382950223588034e-05,-2.22117142947849e-05,-2.8950389338551144e-05,-2.9155980582616734e-05,-3.561041708514295e-05,0.0002997065347102882,0.0006913981698574873,0.000912123496872619,0.0007480439026901503,0.0004116288342714518,-0.0005255885618582005,-0.0016361003379476487,-0.001819727345075191,-0.0015129038486005747,-0.0010257810747740066,-0.00033499328592039376,0.000714729411868818,0.0014078863194464447,0.0016360801541930742,0.0014774731345157301,0.0010271777844828274,1.7668900287756485e-05,-0.0005588349890640352,-0.0008563411583128412,-0.0009437524993565622,-0.0008712127700455777,-0.0003859325969727009,-0.00014189486520806935,-1.5037789852073717e-05,0.00011828095133722604,0.00023411660828125588,0.0003371531945825436,0.0003182910606022173,0.0002234534946305122,0.00013961853864622906,9.344342671462314e-05,0.0002904414792917279,0.00025240292095476236,0.00013008310944300514,-4.704405006884199e-05,-0.0001488305603201083,-1.1738643628190055e-05,7.081924531125394e-05,-2.8623534369758895e-05,-0.00012851099244207237,-0.0001983279340803463,-0.00010465965624779145,5.48119055186155e-05,5.347609185420122e-05,1.007429503731116e-05,-1.76420899636982e-05,4.374121983838939e-05,0.0001773435696440622,0.00011723567613706804,1.6146752937429223e-05,-6.70146434992292e-05,-0.0001060416457132816,-7.028717953354336e-05,-9.198823086452132e-05,-0.00012606490648308286,-0.00013701327509043774,-0.00013225642221492676,-0.00011658837849643203,-7.575816792377832e-05,-3.6697563502227496e-05,7.414846277465578e-06,5.266548344027451e-05,0.00013889106897951493,0.0001088055824303547,7.953093734316268e-05,4.3271195718624406e-05,1.812548938780612e-05,1.9448845097232314e-05,-1.1311616998392243e-05,-3.120939825932414e-05,-2.944883298962675e-05,-3.14393368276307e-05,-0.00010594621944071354,-0.00012175925252616798,-7.705957607816685e-05,-2.739978450834037e-05,1.9605638290571845e-05,1.463431980337809e-05,2.72546020893316e-06,1.890414788341001e-05,3.539421938440054e-05,4.7195401270223495e-05,3.3643232372068124e-05,-2.58251642012624e-05,-1.3860327007511272e-05,3.9155053804319704e-05,6.97102031344605e-05,6.658247199262275e-05,5.213413644335827e-05,-2.3432334583000295e-06,-1.8152997752948965e-05,-2.3483781677201945e-05,-3.7401867693983494e-05,-5.221762500823635e-05,-6.211603001997574e-05,-5.2648112554757406e-05,-2.4500410974819818e-05,5.540265238429589e-06,-4.7463519461088784e-05,-3.3881764068609086e-05,-4.33988342934271e-06,3.583755985234834e-05,8.019975976520898e-05,0.00010552614967869853,9.763212511660359e-05,9.481700125152301e-05,7.492436909861318e-05,5.6512992877077766e-05,2.4221643853113278e-05,-5.998384039847002e-06,-3.6906957227644855e-05,-5.5145883724886815e-05,-6.970930000669014e-05,-9.82830397746598e-05,-9.917206751370638e-05,-7.414698519322739e-05,-4.9270641222630495e-05,-2.880899452834784e-05,-1.9010907982009088e-05,-1.1948539620882839e-05,1.1953092632873222e-05,2.6348521611312597e-05,3.441204141347682e-05,3.236050753441347e-05,2.573216174240491e-05,3.3524989431422555e-05,4.588311503294758e-05,3.560572730640791e-05,2.100213001083433e-05,-1.5276707202760516e-05,-1.715201679238985e-05,-7.314393375443019e-06,1.1455958681765984e-05,3.135765724922197e-05,3.176249144882715e-05,3.500240582470931e-05,4.976490828004239e-05,5.4623694741158817e-05,5.135044418308495e-05,2.050485626010654e-05,-9.553298924255215e-06,-2.731784247304506e-05,-3.192575536633854e-05,-3.1364722014439555e-05,-3.678768259360396e-05,-2.9623158602847785e-05,-2.2157889332921746e-05,-6.5321591005051825e-06,4.543682597994441e-06,-1.1670438435835212e-05,-3.123717821831691e-05,-3.515007687119179e-05,-2.8249701024282904e-05,-1.8973417957693326e-05,-1.3255672703769533e-05,-1.243002352365852e-05,-8.872288643297078e-06,-7.315356744506814e-06,-2.2538063402784312e-06,4.136468415576635e-06,-2.397516991545114e-06,5.836133941897927e-06,2.238980860674324e-05,2.1405587086907293e-05,1.1609269363164588e-05,-4.345376824651117e-06,-1.4821143567227367e-05,-2.4016797946582993e-05,-3.2245860987232526e-05,-3.555894786209208e-05,-3.740797449160198e-05,-2.9355291381791216e-05,-1.7042420029084824e-05,-2.2840365264387984e-06,1.272689852741518e-05,6.646728293817065e-06,2.2916137235158617e-05,3.5961517915948745e-05,4.4207822712874564e-05,4.2713182942797036e-05,2.130675383264336e-05,1.2038957120404756e-05,1.4509558598935294e-05,1.1249439669658159e-05,-9.518100172663876e-08,-1.6501420158927463e-05,-2.3846934951464175e-05,-1.771397213224781e-05,-7.169834231651535e-06,-4.557983685264657e-06,-2.109868070183549e-05,-5.236691042271218e-05,-6.215902178126594e-05,-4.440838190688618e-05,-1.6208021167890762e-05,1.7760479077929616e-06,7.792977720920213e-06,1.5454386445762985e-05,2.1946346856369215e-05,2.9993793728765652e-05,3.839465798634704e-05,3.490655617712548e-05,1.874378319993642e-05,-6.07508015394604e-06,-2.092583521412615e-05,-2.2105714150728794e-05,-3.484718450038578e-05,-2.4209339827808568e-05,-3.351721873094142e-05,-4.977284501588814e-05,-3.9115649482769196e-05,-1.502893742579992e-05,1.790793799284603e-05,5.042890233879197e-05,6.536874617507691e-05,5.787050962356203e-05,2.9093789602462384e-05,1.5874651639084084e-05,2.5088353823240398e-05,3.721287730651787e-05,2.4227094764523723e-05,3.7649065865637814e-06,1.1913198267788105e-05,2.6854068327747873e-05,2.9903988978049182e-05,2.7181615405041223e-05,9.110586655640272e-07,-9.13579669730733e-06,4.730608667975192e-08,1.244500361693412e-05,1.9791249978512215e-05,2.311629730786406e-05,2.6901250897313165e-06,-9.579146232407914e-06,-1.859936829316161e-05,-2.2282193115629612e-05,-2.2664412132557887e-05,-2.107999887431482e-05,-2.5099014057911337e-05,-1.6595326777772447e-05,4.134178275429197e-06,3.1994956034722416e-05,4.116384467927074e-05,2.34250091150843e-05,3.5001966657810063e-06,-1.0215609507395458e-07,6.0478356792302e-06,8.032343811042574e-06,1.6147184703097875e-05,1.1975794258539795e-05,1.2533007887169593e-05,2.4998864483048667e-05,4.234246491775099e-05,3.482612355258481e-05,1.7084438861655496e-05,-1.8014247792039753e-05,-4.856573251010795e-05,-5.60766882926814e-05,-6.25277782061793e-05,-5.983722290068406e-05,-5.168220689141023e-05,-5.01010832100873e-05,-4.651884587796981e-05,-3.186981856395857e-05,6.5570802006061635e-06,4.498165458297643e-05,6.131359523221837e-05,7.747781085898364e-05,7.646190463021283e-05,5.723367946664791e-05,4.045364533485473e-05,2.4723990999091975e-05,3.881418164337572e-06,-1.7462285503513466e-06,-7.312474926294025e-06,-9.786894523698474e-06,-8.336925605840319e-06,-4.417350677566309e-06,-6.032249479710728e-06,-1.3388177973370853e-05,-2.0447084079755432e-05,-2.4066642638633856e-05,-1.8179264806197555e-05,3.5521944744763084e-06,-6.183909470941302e-06,9.669163427616878e-07,1.1196376692295472e-05,1.7280039911134812e-05,2.823341116146342e-05,1.4853025014491573e-05,8.65707096308162e-06,1.2077359228539068e-05,3.258669581787202e-06,1.2680779637936735e-05,7.708666777448078e-06,-1.3723170583043578e-05,-3.608295344106483e-05,-5.119761446734506e-05,-7.312614259221007e-05,-8.834357762561481e-05,-7.741561262327835e-05,-5.836206190203931e-05,-3.0131832037380346e-05,3.203696160951663e-06,2.0346990680269507e-05,3.385356142536836e-05,4.833110141956229e-05,5.0391322602313304e-05,5.180425465328373e-05,5.4829055759994386e-05,6.0599559388100145e-05,5.631935465899498e-05,4.3012137013473166e-05,2.4233679658997512e-05,7.270124741409726e-06,-1.2505367267640377e-05,-3.5207746513147715e-05,-5.253684689192936e-05,-4.685207871788814e-05,-3.887764464491257e-05,-2.9846783357456998e-05,-1.4718716091069102e-05,-6.853361052795274e-06,-8.82791812566405e-06,-1.7046928113713482e-06,-1.800734746623934e-05,-2.0147036283771915e-05,-6.61750068306531e-06,-8.7029428616077e-07,8.015627595667012e-06,7.263866362266962e-06,1.4089022056334805e-05,2.4827363473396466e-05,2.7521102706992836e-05,3.726661284526137e-05,1.748909333758991e-05,-8.685495159369579e-06,-2.160624531049076e-05,-1.8671084336471046e-05,3.265183639120869e-06,6.657258085606201e-06,-5.748523373149056e-06,-2.2675415305712948e-05,-3.075480313427224e-05,-2.6119568445057005e-05,-1.3218356551311475e-05,-1.0395596453798833e-05,-6.543428848349582e-06,-4.633886337826682e-06,-8.540462631728453e-06,-1.793685228853204e-05,-3.308952151036868e-05,-4.596726791148174e-05,-4.704160934273471e-05,-2.84162803892393e-05,-4.7615032127991704e-08,5.499677235395216e-05,9.900861780332312e-05,0.00010437651837495434,8.574632668619367e-05,7.084308709093654e-05,5.9929627061830204e-05,4.562372232313394e-05,1.8391106646548e-05,-1.8102568772108906e-05,-8.884737850460426e-05,-0.00011386303744371644,-0.00011289381574329335,-9.59249793011495e-05,-7.286937970682026e-05,-6.252240472947572e-05,-2.395005993736369e-05,3.230248585440849e-05,8.127273088149666e-05,0.00011802034752903956,0.00011715990922686049,8.030625732386378e-05,3.197793717970054e-05,-7.0370499468624865e-06,-3.312992811804327e-05,-6.266205697745363e-05,-8.49145755439526e-05,-8.976023851108617e-05,-8.621597943846267e-05,-6.933834942033844e-05,-4.440854824936435e-05,-1.57755085274952e-05,2.3235904224276035e-05,4.803018457458091e-05,6.145946908118269e-05,6.655501703944096e-05,6.436307511100667e-05,6.764090984691117e-05,6.609439981344321e-05,4.889905879521647e-05,2.970304032347047e-05,4.666722988911901e-06,-8.497772923593796e-06,-5.046303263952957e-06,5.408840480566155e-06,4.563963794568324e-06,-6.648108280798886e-08,1.1800635726893596e-05,1.9826338827765114e-05,1.1427894650280022e-05,-3.499929666126043e-06,-1.861245721891059e-05,-2.4412690719245342e-05,-2.2970040835942882e-05,-2.264257235163094e-05,-2.568002817354202e-05,-2.185904977088874e-05,-2.3005222007799623e-05,-1.7136175962383485e-05,-5.3877499481744875e-06,1.4879708747773582e-05,3.358904712557681e-05,4.0048687820881075e-05,3.8909138153392985e-05,3.95795131756686e-05,3.1745056895956617e-05,1.4305951557040056e-05,-6.598129263639175e-06,-3.151948806444551e-05,-5.738763201873942e-05,-5.716845541101152e-05,-4.811210689122785e-05,-4.3019996483633004e-05,-2.6363346131639438e-05,-1.3520613688729826e-05,-5.079719004031258e-06,-8.879724309249283e-07,-1.8100263475111943e-06,7.075209138929403e-07,2.043157301899117e-05,3.90884597363879e-05,4.545716607607904e-05,4.894423323242842e-05,4.6935967768038386e-05,3.476830641016898e-05,2.4564314575182977e-05,1.4161235356282798e-05,6.11049476167002e-06,5.861074465466462e-06,1.0921738313685262e-05,1.616525556869752e-05,1.7401492258108182e-05,9.594873242415107e-06,3.264194142890581e-06,1.0046357742363124e-05,1.291642979196349e-05,1.723533565645637e-05,1.9649887254006957e-05,-1.9056439139764422e-06,-2.813395487013551e-05,-4.0185519221446886e-05,-4.964239621384138e-05,-6.181467962515872e-05,-7.78046984924002e-05,-8.252837692327045e-05,-7.221894999518884e-05,-4.6426315003753495e-05,-2.1391090007135397e-05,-9.985353624629037e-06,-1.4538942660996057e-05,-7.968738264539661e-06,3.160650043387794e-06,2.0024376758808417e-05,3.4782186141080875e-05,4.3470860369161856e-05,4.636023665527501e-05,5.087741551391587e-05,4.3550813166949854e-05,4.0588390486325144e-05,3.986342549120288e-05,2.7935504037798986e-05,6.916663324069758e-06,-9.365604732140727e-06,-1.939868838813703e-05,-2.409933047527055e-05,-2.059428538233528e-05,-9.66219796929702e-06,-5.338699384049383e-07,-1.1575134920577913e-05,-3.462075142147284e-05,-3.34227648840906e-05,-1.721927509305633e-05,-1.7446360896528934e-06,1.1903383710054628e-05,1.7779505788060923e-05,1.653906515640262e-05,1.9348696831509892e-05,1.9458039972410814e-05,1.1159767267297515e-05,2.2510464753829273e-06,-4.9607712570932255e-06,-1.8352707761864808e-05,-2.1237807353847855e-05,-2.285008647111125e-05,-1.58486536008343e-05,-7.649880909110863e-06,2.501821314907858e-07,-1.2870989370089531e-06,2.020484110341335e-07,4.514621260962123e-06,1.5267476145141068e-05,3.327150087732052e-05,4.693710146422569e-05,3.999081079950096e-05,1.0088309342702713e-05,-1.589610085838403e-05,-3.533111778450181e-05,-4.183786741882387e-05,-4.359567560617448e-05,-4.7355076987660716e-05,-2.7938174139621914e-05,-3.5733161457135094e-06,1.0393224979616973e-05,1.524280101947576e-05,9.084176469976315e-06,-3.2903320658295564e-06,-9.611801863444576e-06,-1.4157945139325827e-05,-1.7665862723843554e-05,-1.888470346732393e-05,-1.3089300407859905e-05,-7.331105655751111e-06,-1.5058854518625395e-06,-1.880396012511832e-06,-9.21210602540013e-06,2.2609539641445905e-06,2.6120217762585793e-05,3.567209986810334e-05,3.527670995434655e-05,3.501742057462314e-05,2.417496323939969e-05,3.3546938140779546e-06,-1.6033807686482228e-05,-3.1293041194455826e-05,-4.328410596287268e-05,-4.938411030752161e-05,-4.650059613470396e-05,-4.8403119197793085e-05,-5.7146240173260455e-05,-6.377570725163775e-05,-5.557010298805538e-05,-3.823247303352629e-05,-3.598251997507053e-05,-7.075463687091514e-05,-0.0001306661387468691,-0.0002032473955100041,-0.000270871146509128,-0.00032314137719215575,-0.0003571989997217013,-0.000341544749021415,-0.0004142687252884224]

    ACT31_RIGHT_HAND_Z = [
        0.0,-0.0002736209221331637,-0.004758754526029575,-0.008162916037586456,-0.010195772051393071,-0.010847251918745764,-0.010248017702872966,-0.008614018704011859,-0.006358317432226797,-0.003819583643069111,-0.0012967648464158456,0.000943994738905748,0.002785697245759229,0.004085208151580903,0.004721860675840192,0.004722879821174888,0.004189926131187451,0.0033064525025662786,0.0022999094568197814,0.0012617295951783604,0.00022347590192067413,-0.0007089929233172499,-0.0014545680838919655,-0.0018992993593299494,-0.002006802300662566,-0.0018775108745170098,-0.0016146731494129773,-0.0012638418490622217,-0.0008566689717515789,-0.0004387657828094744,-7.715877802393334e-06,0.0003314288068469474,0.0005664914954420145,0.0006807405992366372,0.0006820622108413035,0.0006098065634144388,0.0004954800600782925,0.0003274027002877202,0.00014994827649025968,-3.2696670217659766e-05,-0.00019621560607835787,-0.00030363834575393385,-0.0003680000321510757,-0.00040640872137808143,-0.00039067827286445644,-0.0003638196623958993,-0.0002815164139981071,-0.00018949314532632454,-0.00010203570345275225,-3.55019628325446e-05,3.079833254299329e-05,9.230018599617086e-05,0.00015729035206321236,0.00019050060099006332,0.0002124928864656102,0.0001961851727895432,0.00015794884092708422,0.0001290200798306132,9.828092367143095e-05,5.5082444515635054e-05,1.547861344564534e-05,-5.0624312286799246e-05,-8.893516154702125e-05,-0.00010202766560669327,-0.00010219799126241523,-9.730825644031023e-05,-8.207498491530418e-05,-8.359069217738438e-05,-6.415875533380611e-05,-3.8222133485212624e-05,-1.467786924148141e-05,7.599256611667784e-06,4.7664073326907605e-05,8.803756270403904e-05,0.00012461179459026783,0.00015173178311760557,0.00016308011676493498,0.0001644388552953002,0.0001613230003888324,0.00014620385322469586,0.00012196885982566573,9.639758299914379e-05,6.948949250624348e-05,4.260977362699857e-05,2.0980873886592785e-05,-3.198896959660911e-06,-1.6744779780810765e-05,-2.4679449686836022e-05,-3.62016132993448e-05,-4.6466443725335425e-05,-5.661465671115623e-05,-6.989436700929613e-05,-5.8646008524938846e-05,-4.6816597393077586e-05,-3.6224534224690084e-05,-3.814392585329341e-05,-3.687748560077075e-05,-3.350897201379888e-05,-2.0994245220586396e-05,-1.2805130678672551e-05,1.6974705107109575e-06,-1.3860043708200905e-07,-8.631116028572256e-06,-1.775922304023371e-05,-8.652492842716642e-06,-2.9185503323170962e-06,-2.111267339632847e-06,-9.958452704874042e-06,-1.717933023197942e-05,-2.8107589638559967e-05,-1.5781623497713786e-05,4.557094392802322e-06,2.3792603621931304e-05,2.759201498941745e-05,2.208187014446232e-05,3.203593363370047e-05,5.8169262286917484e-05,7.530664157666586e-05,6.933248191029107e-05,4.966654101357342e-05,1.6147784131930912e-05,-5.654241450911183e-06,-9.281641656369464e-06,-1.575068591048639e-05,-3.281608818570655e-05,-5.5374723080983505e-05,-7.493406891842621e-05,-6.503640814154376e-05,-4.418063464680093e-05,-2.598403935606607e-05,-6.421174472432101e-06,1.3623098130578094e-05,2.7821484161875204e-05,4.9804039263223904e-05,6.904597965673555e-05,7.533096599797597e-05,5.7144233830811796e-05,3.140261946224342e-05,2.659611917106274e-06,-2.1488232882689314e-05,-4.1658751393306116e-05,-5.23713320331326e-05,-6.237588559978234e-05,-7.094815373227872e-05,-7.44116790860314e-05,-5.970993283081075e-05,-3.338725686231884e-05,-7.471863741959725e-06,3.497304862854585e-06,1.1224616388019713e-05,1.545706495114714e-05,2.8675614925982346e-05,2.6999284619980613e-05,1.9890990090660463e-05,2.0916894741701325e-06,-1.9136733209728972e-05,-3.299272617579943e-05,-3.809627680267876e-05,-4.765734826203363e-05,-5.394076666811344e-05,-5.857614136774727e-05,-6.490497343033893e-05,-6.701132875605936e-05,-5.993472463556568e-05,-5.870428598695537e-05,-4.424110003044211e-05,-2.5559207634212524e-05,1.1286472712637797e-05,5.469744208577276e-05,8.848229040592188e-05,0.00010242631878911843,0.00011295958771645437,0.00011484278232741945,0.00011250627103090487,0.0001089135621345571,9.880102127021797e-05,0.0011748823027022592,0.0025889173873165088,0.002994204723366211,0.0025196621979397273,0.0015154344069472742,0.0004564047145291146,-0.0004875202261570308,-0.0014575395040649691,-0.0022327362841526722,-0.0026733063151088924,-0.0027090318299986984,-0.0023995959900337516,-0.0020084150987717017,-0.0014012159330221343,-0.0007165087518824602,-9.823888448272719e-05,0.0002866654642693133,0.0008139503943243176,0.0011369748215522546,0.0012563808637187529,0.0011893059195879796,0.0006749803442882559,0.00020287107146984987,-0.0001287806798702876,-0.0003349035936083844,-0.0004288710724388339,-0.0007702006467779081,-0.0007741732577805817,-0.0006157109183269746,-0.0003474339975135204,-3.7477760998195964e-05,-9.7020734069851e-05,-9.483966627982812e-05,3.8934206468947154e-05,0.00023027876574590242,0.0004222505587026362,0.0003864899410422832,0.00026727886096166955,0.00022054037122584455,0.000224928208040507,0.00024613432400913404,0.0001794409595282413,4.256230567219345e-05,-2.9473077008363462e-05,-3.9873489228850935e-05,5.328042007271615e-06,3.8114328131898756e-05,1.1292319515394394e-06,-3.554859954231307e-06,1.3525742822656039e-05,5.5937344061398665e-05,0.0001025991377526877,9.30956911079368e-05,8.592308829999824e-05,7.948018742453746e-05,8.839699281869508e-05,0.00010947242445463191,8.085886214977828e-05,4.0415212893124894e-05,8.876492776517217e-06,-1.4308815605207535e-05,-2.4541069271492633e-05,-4.776036230565547e-05,-8.160487911592635e-05,-9.510451423254012e-05,-9.860154821433887e-05,-7.955507032453735e-05,-3.399703695024859e-05,-2.6641767690219935e-05,-1.6108337800073767e-05,1.4833062320610512e-05,6.135421741082347e-05,9.74553621627954e-05,9.700746556216671e-05,8.251589723050732e-05,7.434745475421924e-05,5.4912962390329374e-05,2.4568772496432215e-05,-1.3837177293070261e-05,-3.9129139238267804e-05,-4.870070961260606e-05,-6.194282189045062e-05,-7.081673464978483e-05,-7.36505448388258e-05,-7.395365474687978e-05,-6.728327089810939e-05,-5.3981349977036376e-05,-4.044977279216253e-05,-3.0708197370369564e-05,-2.932406987065586e-05,-1.573267927331042e-05,-1.6576357061755365e-06,8.837973424165498e-06,2.216643616369317e-05,2.382503135730581e-05,3.544249489455173e-05,3.243757609245312e-05,1.81759686730397e-05,1.8865175982991583e-05,9.144840086951063e-06,1.537746030668705e-05,2.7380542392726827e-05,4.2853141270834614e-05,6.938700651779723e-05,7.809572015780491e-05,7.659461567197483e-05,7.415329464955749e-05,6.943703136150021e-05,6.288207977736374e-05,4.2436133238186534e-05,1.7747112185817576e-05,-8.754410726207534e-06,-3.169629079133887e-05,-5.1219539210531916e-05,-6.207752459414736e-05,-6.092306078376919e-05,-4.731455180218138e-05,-3.085067355429609e-05,-2.59812556022761e-06,2.808715550385715e-05,3.349559040742038e-05,2.3418721017931732e-05,2.3702614904940755e-05,3.5495703642892255e-05,4.3251004107985724e-05,2.6628682723217173e-05,-3.6752898732517867e-06,-1.4967692981416234e-05,-1.6830709558498413e-05,-1.0370629487368268e-05,-1.9443799467638094e-05,-2.8626757758293755e-05,-3.223988434619609e-05,-2.953181321390699e-05,-1.8711734809033723e-05,-1.6268227103003508e-05,-1.5431738732691716e-05,-1.4618990550388068e-05,-5.03305410543605e-06,9.06966778885554e-06,8.431336944101051e-06,-2.1949874806585513e-07,-7.059556145741975e-06,-4.441851638460701e-06,1.5716916924743923e-05,2.7564327175088656e-05,3.226188094719849e-05,2.7261581792814634e-05,1.6543908613227135e-05,1.0119642656094367e-05,-5.103730106460473e-06,-2.112364165980561e-05,-3.27736715122389e-05,-4.0172904557595034e-05,-3.4792769386331915e-05,-2.618604821808488e-05,-2.313132171359753e-05,-1.8009118704011762e-05,-6.104991910357316e-06,7.859632422643983e-06,2.7330735383702254e-05,4.316713763496576e-05,5.925897063511609e-05,7.287736635518294e-05,6.836605270634745e-05,6.187784187546111e-05,5.058634958330244e-05,3.601108071489505e-05,1.437368857815116e-05,-2.596386235588382e-06,-2.4633680766611e-05,-4.6926715876497646e-05,-7.113912856894624e-05,-8.418677809340887e-05,-8.589745593649542e-05,-7.018252293525173e-05,-7.736629218623902e-05,-8.373731340551077e-05,-8.386653470111069e-05,-6.836634072784023e-05,-3.063562900370535e-05,-5.604331943760151e-06,2.257248733719359e-06,7.449976491035448e-06,3.1276024812668626e-05,8.389002882246344e-05,0.00014456002406783972,0.00018252766648968062,0.00019557773335337992,0.00017737426731325186,0.00014295064139350458,0.00010498270871006345,6.802564598059848e-05,2.165754087079917e-05,-2.866771984580701e-05,-7.066699110403465e-05,-0.00010676746605471869,-0.0001269161334654522,-0.00013376391451176746,-0.00014030924030717433,-0.00013909807098992123,-0.00012276761399705844,-0.00010043069329307453,-7.59657720570934e-05,-5.7002115143513384e-05,-4.3558465099120206e-05,-2.5506957621374944e-05,-1.4811609820313988e-05,-1.579017203996699e-05,-1.2193004199468035e-05,-7.876125596722288e-06,-1.135414698793005e-06,7.896491110280412e-06,1.0650760479840079e-05,1.5000180976479618e-05,3.458074428274015e-05,5.044223887546045e-05,5.950285649743234e-05,6.839431869398817e-05,6.792286004591817e-05,7.025070100330853e-05,7.602498996136195e-05,6.78413065757073e-05,5.931967857006987e-05,4.129407440309333e-05,1.3503943721811849e-05,-8.437896435410486e-07,-1.368707731812747e-05,-3.34365389603714e-05,-5.139436259349244e-05,-6.436996106632263e-05,-7.281800453392392e-05,-6.431320618891339e-05,-5.746727708794562e-05,-4.656260524024756e-05,-3.667687427485448e-05,-2.497026513291802e-05,6.522897067395075e-06,4.8347861209828545e-05,7.908023054437322e-05,9.65319886946044e-05,9.738911783694944e-05,8.264394575958576e-05,7.210458967523414e-05,6.181779095282308e-05,4.432413649194574e-05,1.928151847828895e-05,-5.951419336698117e-06,-2.4176404802246496e-05,-2.6749898901648518e-05,-2.1757040922712327e-05,-2.1014699953860516e-06,1.301650448211905e-05,1.3215985503306487e-05,1.2440374829809223e-06,-7.709085809680945e-06,-1.9148331578624953e-05,-3.522603763876343e-05,-5.522511614351863e-05,-7.569340552470555e-05,-9.070904264101924e-05,-9.307675255219111e-05,-9.731204293034758e-05,-9.592113790926037e-05,-8.795426404806761e-05,-7.442033831816702e-05,-5.7416707604268224e-05,-2.937870091886231e-05,1.360007075512969e-06,2.9575286041772744e-05,5.663095950311448e-05,7.79091212058712e-05,9.809817780145266e-05,0.00010714159058926542,0.00010306457474391566,8.188556570855198e-05,4.809913604884069e-05,7.977654389649632e-06,-1.25483609966252e-05,-2.6206785475720273e-05,-3.271346479279906e-05,-3.859391293407237e-05,-4.448696097884308e-05,-3.60889587338841e-05,-1.167851925036076e-05,-3.376151512895887e-06,4.036262221532036e-06,1.9101436886991623e-05,3.3057918328445153e-05,4.3492990550558816e-05,4.726665342860629e-05,4.041981757290044e-05,3.417036023258292e-05,2.996424424080056e-05,2.3542884297509515e-05,1.4433340729791516e-05,-2.7200941951798804e-06,-2.1181734686713457e-05,-3.0963379271763885e-05,-4.037679971974104e-05,-4.904169201117482e-05,-4.838209737024284e-05,-4.881800139056491e-05,-4.345210054048615e-05,-3.064172876847134e-05,-1.3291351161110753e-05,-7.193209701682139e-06,-1.0723790106927345e-05,-2.0297871682943046e-05,-2.1931654316182593e-05,-2.0376789648279625e-05,-2.548567615722044e-05,-3.789831948587358e-05,-3.982897481644221e-05,-3.605400529899694e-05,-3.6764947569759104e-05,-2.8284106095736904e-05,-2.7529565047722297e-06,3.613132140259982e-05,7.140580729563309e-05,9.85542856468832e-05,0.0001099404998682636,9.874849634327561e-05,7.84984305426567e-05,6.152166398320159e-05,4.1254307260539665e-05,6.379306139190198e-06,-3.0355140480901258e-05,-5.61826992708735e-05,-6.580233493601766e-05,-6.285372924609847e-05,-5.4626201920182325e-05,-4.7680721036908834e-05,-3.92299928684826e-05,-2.5868815149556748e-05,-8.430798703360967e-06,9.912960554927932e-06,1.3047217089066864e-05,5.480783801462593e-06,2.3779742923573732e-06,1.2410007220281364e-05,3.271807838649043e-05,4.469295217190868e-05,4.0722750678314656e-05,3.3417416301841085e-05,2.1787658155414347e-05,4.267047262558451e-06,-9.710367128981789e-06,-2.161978555684361e-05,-3.651160858852083e-05,-5.127545601080569e-05,-6.39830599575579e-05,-6.35287878761246e-05,-4.9890804154209955e-05,-2.2701817336242817e-05,1.117071123856215e-05,3.994985476686791e-05,5.61462489105107e-05,6.988204857395277e-05,8.630933446545612e-05,0.00010618216414476085,0.00010845108932772281,9.553008479408342e-05,7.606861260634374e-05,5.528466383587779e-05,3.6909300228832736e-05,1.1052797182878176e-05,-2.1770673423253186e-05,-5.695354843756751e-05,-8.693968371569421e-05,-0.00010097003092475498,-9.272274653708714e-05,-8.305702422029763e-05,-7.764626886328105e-05,-7.151466102019924e-05,-5.322676425781726e-05,-3.5573516062452896e-05,-2.0446176080979587e-05,-7.796086643925402e-06,4.520210658040465e-06,1.764129634713256e-05,2.2470244688943617e-05,8.007030702279316e-06,-5.616819028024406e-06,-1.2777378025237417e-05,-1.4559796090433364e-05,-1.1922832398592248e-05,-6.907030552745931e-06,1.3365921994097344e-06,7.758420028334075e-06,2.2930638556260175e-06,3.843902304116518e-06,1.2881380004378522e-05,1.8353834041674492e-05,2.7434986568058752e-05,3.6620080237112844e-05,4.300994157115813e-05,3.9341429224200075e-05,3.368307200595607e-05,2.797133185455744e-05,3.280596504814851e-05,3.3843782160426516e-05,3.491435923894985e-05,2.8358873631909835e-05,1.712459825328944e-05,-6.036500244087402e-06,-2.846765254231433e-05,-4.317972704520964e-05,-4.9067379894451423e-05,-4.6311003590920426e-05,-3.754293315294612e-05,-2.114771534307837e-05,2.7867261422807115e-06,2.3283218100716036e-05,3.603164081107451e-05,4.7460745310378047e-05,6.165465448072292e-05,7.082784202238405e-05,7.129334276939294e-05,6.012750171759129e-05,4.596045660900611e-05,3.591481775642108e-05,2.2292847027692085e-05,6.1001972215053505e-06,-7.787891403913787e-06,-2.592843028949514e-05,-4.180711679670735e-05,-5.1949644875255145e-05,-7.115577363276994e-05,-9.580963648238721e-05,-0.00011795051924483355,-0.0001238147871241689,-0.00010917971263617523,-8.350631206899927e-05,-5.9019138289986015e-05,-1.76447044429082e-05,2.0207470436761074e-05,4.5415840991407466e-05,6.44213317001022e-05,7.262457181684062e-05,7.837243548305694e-05,7.903428534351997e-05,8.323539877981436e-05,8.469257475406235e-05,8.0002125034198e-05,6.464641448244151e-05,4.2108883621980684e-05,2.0727494353275353e-05,1.8569712440778337e-06,-9.385557618849232e-06,-7.860471682862972e-06,-1.6286821908136277e-05,-3.199843389584972e-05,-4.762545007155772e-05,-4.602933587164983e-05,-4.532815886904951e-05,-5.2554421826606175e-05,-4.717879620025175e-05,-2.651687978179565e-05,-1.3992613918821274e-05,-7.750696585677642e-06,-2.9539479254532167e-06,5.704816499030897e-06,1.6650201437284213e-05,2.2029547550589927e-05,1.9135659797702797e-05,2.126315944976841e-05,2.6038355913594383e-05,2.535093698440501e-05,8.718050718239347e-06,-6.926334750361577e-06,-1.2687980601830273e-05,-1.9938451827684084e-05,-2.740815407529406e-05,-2.752418144818779e-05,-2.8518614857914342e-05,-3.936012291066538e-05,-5.822215493330476e-05,-6.417161602138968e-05,-4.912392674371549e-05,-4.0613517933538396e-05,-3.8025469625362953e-05,-3.9832960938686574e-05,-4.07773641066909e-05,-2.7355041115616847e-05,-1.3384243699202642e-05,8.062996323681643e-07,1.962808451148467e-05,4.042405242140271e-05,5.497791468750537e-05,6.759179716835937e-05,6.893639733813899e-05,6.08022711735739e-05,5.5859182569373636e-05,5.580279207152087e-05,5.598987487452623e-05,5.2888599606505485e-05,4.2843589192197726e-05,3.461254022387791e-05,2.897139715015695e-05,2.166759769572141e-05,2.0356237186980037e-05,2.3227733109198974e-05,2.2605220117112025e-05,2.6068482934981036e-05,2.8592337021513073e-05,2.3321926242297758e-05,1.3314588270727917e-05,-2.4179105946214823e-06,-1.9726241869035207e-05,-3.2383406721781584e-05,-4.0022131443908394e-05,-4.647963268549283e-05,-4.8182947520355295e-05,-4.578174602331813e-05,-3.836522677218534e-05,-2.8851118075869966e-05,-1.483157916903652e-05,-6.75767282120412e-06,-1.0094526342576303e-06,3.086827858509035e-06,6.453669966658232e-06,1.078852532512313e-05,2.3273718625781212e-05,3.6924745475758876e-05,4.551392558942242e-05,4.455219587975787e-05,2.9765863822386363e-05,1.5468002381553754e-05,-1.5023939042926456e-05,-4.661100244936207e-05,-6.466250748504102e-05,-5.9926329068722346e-05,-5.833345880309367e-05,-5.3877235878916496e-05,-4.349697113746959e-05,-3.580167231929781e-05,-5.3806249312884654e-05,-8.385734241408792e-05,-0.0001207358249822634,-0.00016425070439538662,-0.00022420840066062275,-0.00029644089713906386,-0.0003883871514929708,-0.0004973001241378959,-0.0005832746107844913,-0.0006117828793438248,-0.0005929010424203994,-0.0005524429155038519,-0.0005371084204428854,-0.0005799025227797501,-0.0006613393676676509,-0.0007405731701270783,-0.0007832589882902733,-0.0007808908561665113,-0.0006026067672189193
    ]

    ACT31_LEFT_HAND_X = [
        0.0,-4.507425033843273e-07,-4.506069817398081e-06,-1.3655986990327841e-06,3.1029198371807177e-06,8.575275993910028e-06,-1.3599841371207246e-05,0.00010056270018777734,0.0001943492567975151,0.00023487509244067716,0.00022976555349671718,0.00017491350772021273,0.00010891448378401373,3.9774180926780606e-05,-5.8318341561011265e-05,-0.00014023797841406155,-0.0001989209551101661,-0.00024206514729504004,-0.0002431151385874189,-0.0002214911662127237,-0.0001897894664882246,-0.00014142559213115496,-9.476080501274634e-05,-4.601362569932118e-05,3.051616914821904e-05,8.784405709064351e-05,0.00010739343210699868,0.00011967956864753175,0.00011471236401569025,8.465906835071798e-05,6.487366847312054e-05,3.645623483050362e-05,-5.526458047539773e-07,-3.3880688165485514e-05,-4.6646612426589734e-05,-9.803658688307493e-05,-0.0001263880697025223,-0.00012393337595245486,-0.00010542009638195646,-5.848228793395601e-05,-3.70394706686677e-05,-4.9460807580334336e-05,-6.0396075032244106e-05,-4.588587777794502e-05,-2.1683745968550662e-05,3.096519652277589e-05,5.70557970332964e-05,4.3003733051668474e-05,4.727583996486491e-05,7.736159523729341e-05,0.00010060564636870414,0.00011555649967091629,9.476507960104752e-05,4.455711392880272e-05,8.566427695675405e-06,-9.284521111505065e-07,-6.487948309825135e-06,-2.0402342738886384e-05,-4.336759245534709e-05,-5.152811370814659e-05,-2.934290221404405e-05,4.132705100961447e-06,2.2904359280398685e-05,3.2867399529675676e-05,2.3514265887199155e-05,1.4986745411239925e-05,2.137241206268039e-05,5.5945763350355054e-05,7.495505191578721e-05,8.292775553850187e-05,7.202553471669364e-05,5.0297125015546936e-05,3.0032496646068966e-05,3.261322790036736e-05,2.6308703171967586e-05,-3.0826496844388805e-06,-1.667361823571059e-05,-1.2735324946191676e-05,-1.6459720245848225e-05,1.942465336297159e-06,1.0507623565034812e-05,1.359990389459747e-05,6.317543829357592e-06,7.125345112277199e-06,-8.692814636276794e-06,-1.4869784459442503e-05,-3.392278082260133e-05,-4.649705716606311e-05,-4.1857603770174416e-05,-3.460208905595416e-05,-1.891755715520134e-05,-1.3253230131543635e-05,-4.874235737632665e-05,-4.607965670760106e-05,-1.3168165217179544e-05,-8.41793251492022e-06,-1.0794432343658833e-05,-1.8151722964235183e-05,-2.5584789405766315e-05,-1.8049413709112788e-05,-8.55826991446415e-06,1.7357752866777798e-06,2.2395095309910485e-05,4.0976134015728604e-05,3.591352018387673e-05,3.23700949822066e-05,2.3319846911122095e-05,1.2571560036635285e-05,7.128531319939818e-06,-9.12157423863647e-06,-1.4939026522872496e-05,-1.1410883129651025e-05,-1.9558682252553187e-05,-2.2068612413740006e-05,-2.5677251868746418e-05,-4.187297503496911e-05,-6.549041169190714e-05,-6.782718390734566e-05,-5.144525250750305e-05,-3.1983665114608424e-05,-9.044951291086028e-06,2.6672138170826866e-05,5.066154412290926e-05,5.6080736845041686e-05,5.012271952496668e-05,2.740551006444448e-05,4.439784586790857e-06,-1.7977697140869264e-05,-2.5965153731970882e-05,-3.029880967588375e-05,-3.029592243627869e-05,-2.4526870169084153e-05,-1.518290742158197e-05,-8.430119293571475e-06,-9.564283335423068e-06,-6.29170294940228e-06,1.5053894602205377e-05,3.1595368730095e-05,5.312818531385426e-05,6.398582147456211e-05,6.445796101775643e-05,6.25872743100151e-05,5.478013602761436e-06,-2.301328280920923e-05,-4.9842072912228685e-05,-7.778567664028781e-05,-8.983967858387294e-05,-9.693870777280632e-05,-7.783800816585395e-05,-4.605793838076777e-05,7.291544634706029e-06,4.528088881116904e-05,7.188945477288104e-05,4.328181978521391e-05,1.3819832638531417e-05,-9.133963892823538e-06,-9.928631781224316e-06,-1.0207921288624393e-05,1.146711781493492e-05,3.495756415712361e-05,5.563824144419633e-05,7.904389976088408e-05,7.519541585555686e-05,8.489746879743441e-05,8.47717712280375e-05,6.525531790984325e-05,4.481598401127904e-05,2.436662553605498e-05,8.0954695199173e-06,-4.711234259266419e-06,-1.326376008236101e-05,-4.6259057654726505e-05,-6.871884144166223e-05,-6.155257203718777e-05,-5.249460636331619e-05,-4.9815419564065555e-05,0.00034062708781907154,0.0005672956148533006,0.0004090262403336907,0.00015850994079237321,-5.8299627150523264e-05,-0.00029513723389245906,-0.0006499647229330689,-0.0008617597473339586,-0.0008087180853203357,-0.0005812046788294144,9.845303607329167e-05,0.0015390978717847463,0.0019003563111064335,0.0016371835642939592,0.001057480068499031,0.0005174208952300944,0.0006401469041957878,-0.0003185775366349339,-0.0010180569500783704,-0.0014550404640467397,-0.001630386945153173,-0.001315869182305159,-0.0006753992618042393,-0.0002861305255917092,-1.1419686849202214e-05,0.00017429468892084734,-0.0004757674196303599,-0.0005404515836831841,-0.00039744217755740115,-9.339311645679922e-05,0.0001908169748211355,0.001326706579771044,0.001545806539784815,0.0013575437551444122,0.0009565031610448994,0.00046533250236225433,1.1972749383903396e-05,-0.0003567048366026443,-0.0006454590347141723,-0.0008383467775353805,-0.0009218587841475465,-0.000901801837102054,-0.0007458658650757603,-0.0005013183731418328,-0.00024696974091023154,-2.1948121034883603e-05,0.0001403143387898492,0.00024374569528033923,0.00034467132873957805,0.0003795494309339958,0.00034217517618906823,0.00022957873622914254,2.235517695003255e-05,-3.436022290970878e-05,-7.789147637247977e-05,-0.0001048229454709305,-0.00012004590655599644,-0.00020579676743917892,-0.00015528501348699132,-8.041707075574116e-05,1.2569819728390654e-06,7.629443228154003e-05,0.0001534499755866111,0.0001943724446188487,0.00020432270897675821,0.0001887893003280441,0.00014876370339931267,0.0001754789179620962,9.673011800572305e-05,2.263529749013501e-05,-4.9094813812557566e-05,-9.999268830421586e-05,-0.00026867781901205456,-0.00030368158832168746,-0.00023876277892911964,-0.0001556019167382439,-7.169306807700518e-05,-8.648023320965861e-05,-8.237767068533287e-05,-4.344204506513598e-06,7.504306005474985e-05,0.00014421442648656485,0.00015508985343585643,9.133804276088919e-05,5.94007841472298e-05,4.2766467337858316e-05,3.645116241653843e-05,0.00011217905192356698,0.0003154461892294073,0.00019606819387817456,6.652710135186204e-05,-4.0252629256718916e-05,-0.00010575620035040037,-4.235468594515038e-05,-9.537646028683063e-05,-0.0001293213978434963,-0.00012507396310924355,-0.00010105170108089876,-7.85022567205588e-05,-5.184781438547181e-05,-1.2029565151236791e-05,2.7643312513907703e-05,6.311066183714748e-05,0.00011177321771008596,0.00011398879938326403,0.00010024414103199947,8.349109293685223e-05,5.622031547579728e-05,-6.975105196175601e-05,-0.0001019417151939817,-9.255550376008104e-05,-7.183899221863332e-05,-4.242557432663802e-05,3.500981997827647e-05,6.83452570645946e-05,5.3261853536098113e-05,3.1582478025051966e-05,8.568172933461505e-06,-6.955211822093168e-05,-0.0001394578588494019,-0.00011415170108997037,-7.170954255739794e-05,-3.745861011183655e-05,2.5704827357784957e-05,0.00010306123836104407,0.0001216893155832624,0.00012051962494028834,9.476359561988813e-05,8.040470249553684e-05,9.077981057954681e-05,4.455586955519757e-05,-7.424334199053795e-06,-5.25920317247939e-05,-6.934259076233545e-05,-7.313380951942962e-05,-6.47949334835121e-05,-6.156876946987221e-05,-4.897668761975643e-05,-3.42418470948049e-05,2.5886626657860144e-05,3.631886123247567e-05,2.9454826910070287e-05,1.7702530987033372e-05,6.460781033265314e-06,-3.262702394499096e-06,-1.0946688291116042e-05,-5.721649434437903e-06,-1.816528893108343e-05,-2.098917697764345e-05,-1.6811206839237905e-06,5.479526027809811e-06,-3.006495166691375e-06,-1.2354242069995758e-05,-1.8213030284971762e-05,-8.515884736819836e-06,-1.1168471048102497e-06,-7.921284051627337e-06,-1.4609214255524288e-05,-2.1241554248352385e-05,8.112159532478639e-06,6.436151756551238e-05,4.979924355184668e-05,1.4982312344136715e-05,-7.961867478119559e-06,-3.9848291051732156e-05,-0.00010804531389346924,-9.50792039688955e-05,-7.118685308160811e-05,-4.716297606027548e-05,-2.122255785876524e-05,0.0001545269517893648,0.00012290830147725792,7.187006271225257e-05,2.2845410175530997e-05,-1.5594693320387362e-05,4.4348677678831735e-05,2.853898457084735e-07,-3.897817300918893e-05,-6.283395194938831e-05,-7.29699958553195e-05,-0.00012988433493993834,-0.00010809262940062412,-6.882755045316605e-05,-2.0253240062989433e-05,2.900903929172303e-05,4.3057250126509367e-05,5.3297196322497045e-05,4.2650279934127785e-05,1.9038362272857898e-05,1.3992767685459646e-05,1.53510765274866e-05,3.8265970249244405e-05,6.43802223436897e-05,7.409791414829933e-05,7.041289143830622e-05,4.718833943553005e-05,1.9025535956940343e-05,-1.1288663974286913e-06,-2.322706959137083e-05,-3.3191452110819236e-05,-2.0628864944153006e-05,1.0900470836600301e-05,-3.636217942623278e-06,-2.390441137385678e-05,-5.046105421895145e-05,-6.330741660394578e-05,-0.00011325615100844609,-8.69513073830084e-05,-5.77148421602255e-05,-2.1558286735869883e-05,2.1205954222121574e-05,2.035818105449329e-05,6.735303297413535e-05,9.829226877092679e-05,0.00012640554884620246,0.00012298855858277594,0.0001695293351902391,0.0001361627840599026,8.912650648788336e-05,4.222406347895351e-05,-9.483428617178258e-06,-9.023876317777796e-05,-0.00012389722747831674,-0.00013254991196032283,-0.00012917113664825044,-0.00011061866137891338,-6.250696561554769e-05,-1.84634854813992e-05,9.955268998927734e-06,2.660229880948971e-05,4.402462625748598e-05,4.063272895333462e-05,4.533567399525806e-05,5.819720738226876e-05,5.047555585877237e-05,3.9466592654340586e-05,1.9655946307779903e-05,-1.2643108547542625e-05,-2.6342396567739045e-05,-2.571824079923225e-05,-2.8739937754639044e-05,-3.998565202868349e-05,-4.9987666696249376e-05,-4.829970423746679e-05,-3.0105606105001834e-05,-1.1723907071007228e-05,-2.4875613453799777e-06,6.778774548151101e-05,5.8057035682116445e-05,5.0240734816069524e-05,4.255460134262184e-05,2.2517713089566377e-05,-9.941884075629873e-06,-1.616955675166936e-05,-2.272913927446001e-05,-1.2067857351072826e-05,-3.626469164958343e-06,9.794816368128127e-06,1.7597852384082406e-05,1.0340872175796873e-05,2.0300855841299582e-07,-1.0946362657520366e-05,7.279989596162972e-06,-2.074048008690008e-06,-3.7218653243519826e-05,-6.620744269108993e-05,-8.620725924263945e-05,-0.00011821570505973074,-0.00012947560345386009,-0.00011827712721004171,-0.00010875108247330881,-8.647979597742907e-05,-4.3217138862800344e-05,1.3279491496788983e-06,4.053616981143519e-05,7.509687686778884e-05,0.00011306481073909913,0.00014135135363903237,0.0001527523121427142,0.0001602917815355538,0.00015174124847697022,0.0001352424293441213,0.00011620834408463997,9.574358142416324e-05,5.131855145500921e-05,5.560002004621679e-06,-3.4380107387357036e-05,-7.232711236608616e-05,-7.195655277298555e-05,-8.035568910697928e-05,-8.417193003498904e-05,-8.417934490002188e-05,-9.290733887254515e-05,-9.616054206498845e-05,-8.836570835242482e-05,-7.144887486265907e-05,-4.546939747791567e-05,-2.535094465993279e-05,7.801129038489972e-06,1.5055187593649711e-05,2.1211107360454403e-05,2.142838116544324e-05,1.5305907259885935e-05,2.290460423348936e-05,2.8206372183124805e-05,2.310884244020846e-05,3.1700451436462815e-05,4.9930469915397126e-05,8.601681972819544e-05,0.00011864855682713233,0.00010900096123086577,8.670111012386873e-05,5.764888906538752e-05,3.257913680893108e-05,7.068864366017075e-06,-1.480010489903984e-05,-4.004463828260635e-05,-6.071524034181528e-05,-7.109669910129697e-05,-4.988729112110736e-05,-4.0097748997205565e-05,-2.9808088684658883e-05,-2.1844670969395932e-05,-2.235353104088556e-05,-1.325438302031639e-05,-8.544159089479224e-06,-1.2630862130541244e-05,-1.8144759290910376e-05,-2.314209360661142e-05,-4.808631310125736e-05,-3.596719932202736e-05,-3.0583963133807824e-05,-2.275308664676002e-05,-8.662754266515587e-06,1.2099957836559483e-05,4.9249867529944795e-05,6.704124980087321e-05,8.147321091282542e-05,8.193775164995416e-05,6.314982401063004e-05,4.003691253983779e-05,1.0095534642449337e-05,-1.6587200234828056e-05,-3.509922557948915e-05,-5.091797532676657e-05,-6.000362384497571e-05,-6.302376260723998e-05,-6.644775203862397e-05,-6.787125300069951e-05,-5.394377108501764e-05,-2.1610723231588565e-05,-7.400839174078605e-06,6.222162601927724e-06,1.7012069381679017e-05,2.1159769546449976e-05,8.701988003329527e-06,9.636647262531381e-06,6.958493528540558e-06,-5.6620498628577886e-06,-2.224873862208848e-05,-2.6946604482154066e-05,-1.9152854041691153e-05,-1.1384880041663969e-05,-6.016186489270216e-06,4.235826786380925e-06,1.3964959145081721e-05,2.5880887684684574e-05,2.9399473310567393e-05,2.8304345600579288e-05,2.363162307406841e-05,-5.9194873393671775e-06,2.419979908048687e-05,5.3134235778248233e-05,6.314096187508977e-05,4.931101352859501e-05,4.167496669646466e-05,2.6016446709439434e-05,2.082268696423756e-06,-2.493418555294678e-05,-3.308184542263546e-05,-2.2623502006339052e-05,-1.6204393894492296e-05,-2.2082042447574508e-05,-3.3854801615798695e-05,-4.647697183500802e-05,-4.530739566338896e-05,-4.076491070649911e-05,-2.6696552584276317e-05,-1.4783431281643789e-05,9.837271873231363e-07,2.1939620242942907e-05,4.969815823754933e-05,4.946615050000505e-05,4.4286506892875366e-05,3.605006764857814e-05,2.178283206402155e-05,1.685034367597049e-05,-2.24357777239268e-06,-1.750554205795477e-05,-1.9898546968441834e-05,-1.0867168928618826e-05,-2.303851114435042e-05,-2.2845934714773957e-05,-1.7011717850651457e-05,-4.3614767597438395e-06,1.4045252897866372e-05,7.477649407303749e-05,7.787295823280349e-05,5.598182568493484e-05,2.880136137212986e-05,8.988563237430792e-06,-8.063857452636042e-07,-1.360532197248479e-05,-3.9512717952871535e-05,-5.1117102536808226e-05,-4.764794019681534e-05,-2.9192390782644648e-05,-7.686313189274969e-06,1.4836253574145185e-06,2.153115432138867e-06,1.5080554201634217e-06,-1.856835766215191e-05,-3.8191800843773696e-05,-3.77611553289151e-05,-3.9234770583891236e-05,-2.6917970238464928e-05,6.74789081187078e-06,4.2610015374471255e-05,4.2266378527370837e-05,3.0400967737337185e-05,1.0480989607106628e-05,8.185979262143399e-07,1.3567208409406048e-05,1.684656327909527e-06,-1.1460841915355015e-05,-2.510379170674417e-05,-3.507273701940977e-05,-6.724898659977812e-05,-6.170824886017426e-05,-3.9854176952514844e-05,-8.463572271554162e-06,2.157326659872966e-05,6.683960370432817e-05,7.738739118831167e-05,7.890927896267773e-05,8.05101904544211e-05,6.79416518862664e-05,5.232374024803396e-05,3.58941657076744e-05,1.5248518699984652e-05,9.237582567556878e-06,1.2866482081752693e-05,5.267081151031403e-05,4.8463620860147155e-05,1.8994399267708645e-06,-3.964871584725913e-05,-6.62142474097325e-05,-7.591519566407962e-05,-8.031687688622473e-05,-8.913385951733512e-05,-8.678076089644314e-05,-7.359564813908071e-05,-4.907104765518695e-05,-2.1180242415051635e-05,-8.820077678580026e-06,2.9005283129662624e-06,1.2045860703629326e-05,3.9567076636119525e-05,8.464225832873498e-05,9.164698179575317e-05,8.875991548972526e-05,7.811435110937827e-05,5.9061569831090465e-05,1.0441952490624437e-05,-7.674940794301387e-06,-2.285836088118248e-05,-3.5187277642549116e-05,-4.084969359301378e-05,8.230971514945864e-06,1.0223105108202444e-05,-5.539324062657385e-06,-2.467455784474848e-05,-3.6249819763061975e-05,-4.4191959881651144e-05,-5.5085781951978725e-05,-5.969513045416745e-05,-5.717297247103336e-05,-4.248929342946028e-05,-3.221468124117234e-05,-2.6348906191406448e-05,-2.5943954215160817e-05,-1.9490041132171357e-05,-1.1151615260763991e-05,1.1537802847076805e-05,2.344030367239134e-05,1.161368295925493e-05,-3.7201845674839117e-07,-9.474089156448267e-06,-3.144418014728868e-05,-4.836323895168293e-05,-4.695132473726824e-05,-3.8783564020975545e-05,-1.2917968450700017e-05,1.8510054362330932e-05,5.6921384746564956e-05,8.056274780232502e-05,9.197830710482416e-05,8.530859607531944e-05,6.921232102977724e-05,5.2593742640761355e-05,4.450795329315847e-05,3.6971630939825114e-05,2.1921769440093707e-05,2.58330563313747e-06,-3.335943866454038e-05,-3.603676669209638e-05,-3.6058731078279396e-05,-3.227629395772772e-05,-2.956296137816539e-05,-4.4936085553860584e-05,-4.779358561016411e-05,-4.0176627275700176e-05,-3.5089959968268616e-05,-4.34451584972141e-05,-4.6895803723210704e-05,-3.908889007158132e-05,-3.431985905735651e-05,-3.377879811345009e-05,-2.7155253409904504e-05,1.844301045333977e-05,5.5700243652415076e-05,7.056201841552624e-05,7.426254138000411e-05,6.746618016167468e-05,7.231024710306244e-05,7.730330305522046e-05,6.078511610607757e-05,4.8337932445713566e-05,3.1137065525894616e-05,1.4280613022706187e-06,-2.6030756944806487e-05,-7.943656373648911e-06,1.0668052320988403e-05,1.7221587120587828e-05,2.6581990842155372e-05,8.04962793294788e-05,8.605590865315368e-05,8.74364917507401e-05,0.00010408991905166813,0.00013196024851446766,0.0001400902202963954,9.630366176353396e-05,7.617975176523545e-06,-0.0001019571345047017,-0.00017435352366747183,-0.00016361905001139134
    ]

    ACT31_LEFT_HAND_Y = [
        0.0,-5.256411175721077e-07,-1.5496687144513095e-05,-3.171041727089393e-05,-3.0406803728096782e-05,-2.5815512854440796e-05,-7.170079479145126e-05,-0.0002649462799756642,-0.0002784426705207019,-0.00020131398110655684,-8.763855157706232e-05,2.0165031512512974e-05,-2.3855784184164416e-05,-0.0001925808244299199,-0.00012720853916789232,1.9720754350939465e-05,0.00015834664837551207,0.00023460389918449008,0.00010368474599244408,-9.519872930822063e-05,-7.115572500989911e-05,3.6170109449781304e-05,0.00013787898958818152,0.00015445635764520897,6.748426355375315e-06,-0.00013953849888437343,-9.712099124983587e-05,1.5024139208947096e-05,9.716446477633585e-05,0.00012173544066724774,3.661847668691061e-05,-5.1014023424267344e-05,-1.569577042909836e-05,6.549691058984856e-05,0.00011007928821929161,0.00010653777690304029,7.947491875125173e-05,1.5366089882163546e-05,4.682763230693431e-06,1.1114593276862286e-05,2.6979938554590665e-05,2.619500847268527e-05,3.922796518735971e-05,1.8090754034835948e-05,3.202768404543036e-05,4.704469541167952e-05,4.253937320621606e-05,4.6601254052281446e-05,5.1173427287541076e-05,2.7294505684377486e-05,3.5408654692632935e-05,1.697199398714332e-05,-1.0132563644257999e-05,-5.112821817578454e-06,-1.1285511244484375e-05,-3.784436933100681e-05,-1.3144354715875484e-05,-1.9461546690433427e-05,-1.985221590759101e-05,1.2780849781437375e-05,1.2418021371502557e-05,1.403887656443458e-05,3.846302399440176e-05,2.7883235452713865e-05,-4.845303179432095e-06,-1.9122856870018422e-05,-2.652599377998161e-05,-2.4079696795822912e-05,-1.0188549258325228e-05,-8.81675234049707e-06,-1.852970145005823e-05,-1.565186488317195e-05,-1.7112348230320825e-05,-4.69071892625667e-05,-4.788925809905531e-05,-5.1807082626159834e-05,-5.739720306315147e-05,-5.622717577054383e-05,-3.323240072227336e-05,-3.507613916248828e-05,-1.2668639095198795e-05,-1.033564771296256e-05,-1.3676719963156693e-05,-9.277995845975874e-07,3.820431949414873e-05,4.987201884864334e-05,5.515228292179923e-05,3.2868472001052166e-05,1.127514239012188e-05,5.3728288035249826e-06,1.4542620445935646e-06,-1.045145663857007e-05,-1.6566296803166734e-05,-3.022967554237341e-05,-2.525049969401001e-05,-5.459155131289238e-08,1.373021168798407e-05,1.1956314838986249e-05,4.5796527453229715e-06,-3.3147996998868065e-06,3.4221677660870256e-06,1.8357992238085848e-05,2.626713003258999e-05,1.3522201068616723e-05,-1.0006043347745675e-05,-1.2687105268641315e-05,-9.365177664332012e-06,-7.728906322156118e-06,-2.8572494613806567e-06,1.2150461700425835e-05,-7.222056922981209e-07,1.406767017606542e-05,3.535323113774346e-05,3.964263776790756e-05,2.2840349762063016e-05,-1.1908453431795929e-05,-2.940767589647247e-05,-2.366010126219915e-05,-6.676025947911603e-06,1.3738196210184842e-05,1.863358463574683e-05,-4.944112483973566e-06,-1.6781146180707114e-05,-2.1281434010246075e-05,-1.4521903376223897e-05,-9.111966346974128e-06,-6.648245657147191e-06,-1.066578956326846e-05,4.763815083875558e-06,3.079066945029864e-05,4.728436480252321e-05,4.8100297694193844e-05,3.043387838788361e-05,1.1835189163069478e-05,1.2064982489489842e-05,1.3664104420485059e-05,1.6882676957626523e-05,1.210704573251164e-05,-5.361832010287539e-06,-3.773721667481542e-05,-4.8964786496372126e-05,-4.203788945739365e-05,-1.8885657235092734e-05,-8.421007320528833e-07,3.582093578961218e-06,-5.122350726356538e-07,-2.051384146737625e-06,3.419161311812045e-06,1.1758914202546948e-05,2.3259100707952084e-05,1.3814789278261182e-05,1.590999439375494e-06,3.4883248027694423e-06,7.547104563734472e-06,1.2840992564777738e-05,1.3434501828519585e-05,6.019658323889085e-07,-1.161348585395426e-05,-2.655298965014503e-07,1.0665245920762997e-05,6.963005282446909e-06,6.398916504421813e-07,5.214830856976211e-06,4.277058092432756e-07,-8.870196379318541e-06,-2.2625706144224184e-05,-5.190279359364986e-06,1.911376014749961e-06,9.493645479557806e-06,7.393914007803707e-06,4.413241122133558e-06,5.689306308199839e-07,-4.924780533201075e-07,-1.3429482285375822e-05,-1.891233391901518e-05,-3.249694185801335e-05,7.088168104866156e-05,0.00028246532272114026,0.000622303751992727,0.0006399545886117215,0.0004536838101230187,-0.00023811827883510678,-0.0010994056128931294,-0.00132430443722652,-0.0011571877136448265,-0.0008219629617754764,-0.0005552997686312657,-0.0003712899976054623,0.0003646807050135216,0.0010028158311598083,0.0012593255920620384,0.0012130615173049735,0.0008023414876926614,0.00029293795688642275,-6.745478770140602e-05,-0.0003102537504348554,-0.0004492767000875034,-0.0009902532569057742,-0.0008832644294046368,-0.0005992870391661739,-0.00023579907399232398,0.00010183709854831885,0.0008125547591171573,0.000960943421287678,0.0007938098338566116,0.000560415737080249,0.00034587532641075806,0.000356559347991449,0.00013814954060678594,-8.914791357893623e-05,-0.0003234309880585997,-0.0004469456174499703,-0.00034513010698513546,-0.0002542899048483585,-0.00027336870537583214,-0.0002677729623385365,-0.0002323302657755837,-6.25902180281643e-05,0.0001346930598669499,0.000159301214916279,0.0001354092772071987,0.00010742311300209643,0.00011963089165089817,0.00018638336657665885,0.00011424378719871229,6.167633332780427e-06,-7.682138984462705e-05,-0.00011105014881355962,-8.541844937339611e-05,-0.00010100116778929035,-0.00012355623650353866,-0.00011985762533999715,-0.00010158051913439915,-5.682674272615966e-05,-2.1722599239835995e-05,1.7228480214451722e-06,2.6546571776911513e-05,6.241411189673557e-05,8.422678361576527e-05,5.4609663130896506e-05,4.421087995614664e-05,3.0248592129853812e-05,1.9986783104204846e-05,4.6411573218296554e-06,-2.1499011785384655e-05,-3.555360673243811e-05,-2.9800703236045225e-05,-2.8577059263951566e-05,-6.271124341237571e-05,-6.991642299302575e-05,-4.7375003606024306e-05,-1.626541146391626e-05,1.8237562438450236e-05,1.3110168841617925e-06,-1.4041225743442328e-05,5.060065003992386e-06,2.12664293370408e-05,3.24503191602111e-05,4.735702406008595e-05,1.6519134174422338e-05,2.044087676904178e-05,6.218479646035023e-05,7.453132229713048e-05,5.3330517502667305e-05,1.212016457641625e-05,-2.6290841453153528e-05,-1.6515222777289075e-05,4.774360887109826e-06,-7.3457827459058615e-06,-4.591043885781252e-05,-3.9881700690338027e-05,-2.194039273610015e-05,1.2872880545343883e-05,3.9983697646328375e-05,3.2434927648324714e-05,2.9060768249936743e-05,1.7729146363622884e-05,2.021331098597563e-05,3.3739397150625026e-05,8.176494148266715e-06,2.2617247334022895e-06,8.407200797524315e-06,-5.063450890812996e-06,-2.868942152150113e-08,4.621179004537425e-06,8.16573970530399e-07,-9.062791651738752e-06,-1.4230006113340014e-05,-1.9740649254819023e-05,-2.6122590953671646e-05,-2.0252746059208768e-05,-8.047219476233504e-06,3.387325168689295e-07,5.507818714108711e-06,3.201273408149293e-07,-6.893974031679725e-06,3.7486948719200225e-06,4.752600137100511e-06,-4.066053689842876e-06,-3.040345582007956e-05,-5.089981198281736e-05,-3.8560318634222824e-05,-1.2586598559550833e-05,-1.0621117150420663e-05,-7.193304423261342e-06,-2.4740737077433577e-05,-6.9617561970125905e-06,2.0933420211948694e-05,4.900662636405145e-05,6.429639946890035e-05,6.273140238654511e-05,4.624648389804643e-05,4.318057402113852e-05,3.553212918732851e-05,2.2974202113130754e-05,-1.9063870380789137e-05,-3.712931138338077e-05,-4.360230562765471e-05,-3.2788838014386574e-05,-2.0770574111054862e-05,-2.4391241595670164e-05,-2.0128196412208523e-05,-1.080121374680775e-05,-2.713531783662558e-06,1.2576975284712072e-05,1.8328631087190787e-06,-1.5771326064880417e-05,-1.9480114780043546e-05,-1.7366421122161407e-05,-1.6419852290779208e-05,-1.1136638585069909e-05,-9.479729750622957e-06,-7.010544403509089e-06,1.6166979730236552e-06,1.3169478274160958e-05,2.381706581593032e-05,2.3389065563562282e-05,2.4084986129571567e-05,1.8832432198153743e-05,1.4307158417207673e-05,-2.2848951483785345e-06,-4.728813968989782e-05,-4.719275234400896e-05,-4.390407408595467e-05,-4.124040313478655e-05,-3.5185853528205195e-05,2.8103288346781877e-06,-1.1070913110339289e-05,-1.719509516449544e-05,-1.1604786207793507e-05,1.4896398379039058e-06,2.189167457764838e-05,2.6857795739140196e-05,1.936292190770963e-05,1.5651927161396943e-05,1.693375824194471e-05,-1.826584624314091e-05,-2.238063674822484e-05,-5.629727914400902e-06,1.1546170147789354e-05,1.8276410222947204e-05,9.23923638198829e-06,1.3380334256691542e-05,2.9034945044228234e-05,4.333416164842392e-05,4.0517894491778483e-05,7.427731328241732e-06,-4.3418952806306926e-05,-6.711254468036395e-05,-6.033507625481411e-05,-3.947339230104796e-05,-2.9929996085684666e-05,-4.050664013036758e-05,-3.473148728293249e-05,-1.4306939921253286e-05,1.1584201861615532e-05,3.955685790588737e-05,5.3096790490582444e-05,5.560951634398903e-05,5.384657431306558e-05,5.717534105815746e-05,6.354968562106838e-05,4.066335035550437e-05,2.8692421026660346e-05,-8.46468675068748e-06,-3.656705083615314e-05,-3.625506027586285e-05,-4.139894665876329e-05,-1.5855936960427674e-05,1.5540428217416808e-05,2.770893433254276e-05,2.0391956740077745e-05,-1.0354934481971469e-05,-2.678365564013713e-05,-1.630953316167259e-05,-1.5113339682423901e-06,-1.365888266210053e-05,-3.614021773159718e-05,-2.7165046686383314e-05,-3.7944984394905344e-06,1.2569748401771024e-05,2.0519569683792552e-05,-4.623737646622633e-06,-1.227186039717418e-05,4.219757258600526e-06,2.5519897363757707e-05,4.253292111272442e-05,4.147978300628883e-05,1.0110907303416872e-05,5.250586278484629e-06,1.1107552052934635e-05,2.1210367350152056e-05,2.204499850102105e-05,9.854259301725602e-06,-5.731875582585233e-06,-8.707416683820684e-06,-1.0912036964366206e-06,8.255887831954293e-06,-8.308600117920812e-06,-2.665304197676785e-05,-4.792243496171771e-05,-4.5672804329613044e-05,-2.4560587542046515e-05,-1.1686659344202465e-06,2.439984480496922e-05,2.791895489861497e-05,3.0891758010206835e-05,4.403215098139614e-05,4.223275615317655e-05,4.70951315941892e-05,4.672104984156403e-05,2.9112996127011916e-05,1.5081364002710596e-05,-6.947296907337901e-06,-1.0426242169388065e-05,-6.245759338124781e-06,-3.090150827003326e-06,-9.770809116801863e-06,-2.827572110626365e-05,-3.073581255190682e-05,-2.207032179703746e-06,2.3193629089622337e-05,2.0884191565622765e-05,2.1351877049493285e-06,-2.3772700572967072e-05,-3.657605534402209e-05,-4.0198914331870216e-05,-3.0319471654774246e-05,-2.904211842704627e-05,-2.5804261355933595e-05,-1.1416607648156158e-05,3.3228364583726632e-06,2.6333444367680968e-05,3.8787355052184274e-05,1.5248690550822945e-05,1.3427426988651393e-05,1.4097069354292041e-05,1.353597215638205e-05,2.176148458021779e-05,2.2189536867483098e-05,2.005755180723605e-05,2.4461060136517143e-05,1.977626382772421e-05,2.7845946107966655e-06,-1.8848036316270517e-05,-2.73887742781977e-05,-2.2658018840430774e-05,-7.151226276107541e-06,-1.4427346387779697e-05,-3.6790613856805595e-05,-3.342492242323964e-05,-3.1525064876841055e-05,-2.673932701347769e-05,-1.381061449836597e-05,-1.8526670842409118e-05,-2.0800624690908642e-05,-3.466327100228856e-07,1.6414901402134154e-05,3.2549810833243846e-05,3.2870898009210486e-05,2.0289351923130603e-05,1.550157494729209e-05,2.0155583385407815e-05,2.1019439924302648e-05,2.8359864534836245e-05,4.0926774011367234e-05,4.939872154279247e-05,4.808824464579356e-05,3.7249925938842177e-05,1.748189217762227e-05,-7.017181280043471e-06,-1.4051430710432214e-05,-2.709077703766819e-05,-3.813868369637622e-05,-3.312316985477719e-05,-3.317733172017659e-05,-2.191403768330078e-05,-7.882717132296937e-06,-6.567018877136966e-06,-1.840022059449745e-05,-4.097570167642929e-05,-4.941889124479205e-05,-4.6712639374483875e-05,-2.44890524061826e-05,-4.170543227117902e-06,-1.1584627912472012e-05,-6.481403187524495e-06,1.1231004022576278e-05,3.180793334450011e-05,4.2520379521579514e-05,2.828231409552631e-05,4.206172899368654e-06,-5.349226782587406e-06,-4.987374045164102e-06,2.734502580490706e-06,6.652411754621715e-06,3.7048224102670405e-06,-4.67843702057106e-06,-1.9187460601542205e-05,-2.366748822631565e-05,-2.9748069121521987e-05,-2.529432341451506e-05,-1.842149870315035e-05,-1.9407034510187123e-05,-1.4369574098336862e-05,-1.4194931185049846e-05,-1.7222736372334903e-05,-1.9110592048788835e-05,-2.492905548888107e-05,-2.423022501209829e-05,-7.838407831493108e-06,1.745384119712469e-05,5.281591947958281e-05,7.593338607358484e-05,7.227796801762526e-05,5.125131445834457e-05,1.3444237245720143e-05,-2.640519491183557e-06,-1.3657896130683019e-05,-2.866602003963095e-05,-4.827404981419385e-05,-6.254971973872635e-05,-6.531739424835708e-05,-5.590108418672293e-05,-4.1450873111842806e-05,-2.6211315017086835e-05,-2.45886210416409e-05,-8.242994439050168e-06,1.7721786832005006e-05,4.627639960663009e-05,7.96147625902641e-05,9.458250221567518e-05,7.085558757256414e-05,3.242941724131718e-05,-3.972495906541137e-06,-2.230695009342573e-05,-3.1456892470664494e-05,-3.9175131582409977e-05,-4.045666478543432e-05,-3.49762519020229e-05,-2.5503770406419665e-05,-1.0792512967093613e-05,7.175695575247443e-06,2.881942751954977e-05,4.436204326013985e-05,5.1404444957268253e-05,4.2453548567388334e-05,2.6646537833385635e-05,2.795738818842618e-05,3.155539716710725e-05,2.688703212831936e-05,2.7155353929333398e-05,2.2360478148293823e-05,2.2950099258915783e-05,2.7333466742730932e-05,3.2249202469376804e-05,2.497746778688445e-05,5.204312287165982e-06,-1.9110480834002604e-05,-3.8445152985494216e-05,-5.7166427088827325e-05,-6.359994647933245e-05,-6.824436375814754e-05,-5.8630684557906474e-05,-4.2298668954461324e-05,-2.0134816375773505e-05,5.770950174554196e-07,1.851271644566603e-05,2.5101431872542433e-05,3.776621460210709e-05,5.1961671937689225e-05,6.763797616344388e-05,7.154145615249713e-05,5.452712490654758e-05,2.6581685723828304e-05,8.137890576025613e-06,-7.001172566742773e-06,-2.439958852241064e-05,-4.141211556258015e-05,-5.142985459684707e-05,-5.326291788897581e-05,-3.2848879474881947e-05,-1.4682434155353556e-05,-7.850360679416597e-06,6.319316610974511e-06,1.8811164463660277e-05,2.689941108989823e-05,2.7299364584855694e-05,2.075598710953383e-05,2.000336718379748e-05,2.4837431440449305e-05,2.2581669933473343e-05,1.1099675284112506e-05,3.3374019290821077e-06,8.50975059000892e-06,5.744477293038968e-06,1.272834418787296e-06,-4.612055899951106e-06,-7.465678829212307e-06,-7.028392657592822e-06,-5.200392979782283e-06,-3.4035142821495128e-06,4.684210932567979e-07,4.4438636417941446e-07,1.241165193721896e-06,8.121428095805862e-06,1.3128186648046501e-05,2.4726837889312844e-05,3.359169476197644e-05,1.7720836767834652e-05,-4.010928318199591e-06,-1.9870264260597855e-05,-3.768028343720102e-05,-5.299909877651507e-05,-6.635852737317323e-05,-6.845923743483815e-05,-6.16468283570043e-05,-4.098144567676711e-05,-1.824683196968538e-05,-1.1685725547263451e-05,-9.52066876325377e-06,-3.174938972620298e-06,1.3783719451495187e-05,3.284915231055183e-05,4.5563234502672533e-05,4.5352523695393876e-05,4.176513137946335e-05,4.388592380462545e-05,3.797664979534847e-05,2.066540164653441e-05,1.993256262418324e-05,1.2312463663838598e-05,3.4788180277809994e-06,-6.528633779392576e-06,-2.5152786577023617e-06,3.0113894667055088e-06,3.828588794119233e-06,-2.4335340525804305e-06,-9.904466439472259e-06,-3.387488047849629e-05,-5.495363558524522e-05,-5.0012076088232816e-05,-2.943251544906063e-05,-1.6382144628679536e-05,-5.49827774065439e-06,-4.7398934627051006e-07,-4.992441756109642e-06,-5.369069604304733e-06,-2.849409507169385e-06,-2.744536754356383e-06,-7.995252449745045e-06,-1.2940457187930502e-05,-1.836805986206526e-05,-1.39149760161649e-05,-1.1382702951574891e-05,-1.2085998504123585e-05,-2.7313633080186895e-06,1.018426939109039e-05,1.3612164886448216e-05,2.027808971771385e-05,2.679611056119492e-05,3.420277923143592e-05,4.7591361729631606e-05,5.312634967878963e-05,3.511790979999065e-05,-6.346440276493491e-07,-2.4762420733834417e-05,-4.178246849681878e-05,-5.0903112042299216e-05,-5.345262228737603e-05,-5.580156280338144e-05,-3.896710465748436e-05,-2.14709158558958e-05,-7.083220306064949e-06,1.5707204312133618e-06,8.220017807870627e-07,2.2239997031300557e-06,1.0131518979456076e-05,1.0557636277703054e-05,1.0922151884830318e-05,1.174061946986693e-05,2.0099860255857917e-05,2.8962618218939903e-05,3.0867946188934015e-05,1.3372616649326631e-05,6.579749059067114e-07,5.8968047201729465e-06,1.7784557243553006e-05,1.793163065795982e-05,1.458632372754006e-05,6.423627623665269e-06,-5.50440556596467e-06,-2.4746203550857088e-05,-3.988419464971183e-05,-5.142371259960449e-05,-5.80623283481552e-05,-5.9550561718808e-05,-5.6573548911906605e-05,-6.171672561816503e-05,-5.8638321646784575e-05,-5.294074306560019e-05,-3.4384912467656414e-05,-2.443404939084465e-05,-4.154616562521949e-05,-9.884042454167843e-05,-0.00017385063916228085,-0.00025361662101939587,-0.0003131248958877772,-0.00035532245898390555,-0.000376205223904291,-0.0003676821502166583,-0.00046715872610239096
    ]

    ACT31_LEFT_HAND_Z = [
        0.0,-0.0002743812128626113,-0.004769243497763443,-0.00818155723201494,-0.010220487282308113,-0.010877606611486727,-0.010288348033893282,-0.008656351357815685,-0.006389953161684354,-0.003836866859019697,-0.001305506921801848,0.0009419090991964017,0.002793136955842651,0.00408996639552034,0.004721296064036627,0.004724389234068573,0.0041985523508103455,0.003321126442720746,0.002320184707692049,0.0012897045964749515,0.0002645162452217087,-0.0006621360932258549,-0.0014080278591747885,-0.0018447160349968105,-0.0019711344906746587,-0.0018631224602244144,-0.0016112888604171954,-0.0012796182215447196,-0.0008752744997783834,-0.00045913722673600853,-5.555473425405007e-05,0.00026888193366212586,0.0005080012483277891,0.0006352260393647608,0.0007002988831386578,0.0006316692112533752,0.0005007441088660408,0.00031526633759765266,0.00015277416327881814,2.3402930372688166e-05,-0.00010516484962207049,-0.00024210563472742565,-0.0003322146345755429,-0.0003915535880502185,-0.00035907957206271203,-0.00027242712186473087,-0.00019663034448489755,-0.00015537806942163455,-9.742197157269587e-05,-7.109033125167492e-05,-4.090219206154097e-06,5.33323169843238e-05,7.875591367061553e-05,5.624071679867972e-05,5.314676508346843e-05,8.683635202695947e-06,5.866704768676602e-06,9.828255392077705e-06,7.776361668418751e-06,-1.7538497265822436e-05,-2.230433870635861e-05,-3.7219589429205515e-05,-1.3756658092625496e-05,2.642130265245903e-05,4.501375086545475e-05,5.919307885732447e-05,8.762474656569899e-05,7.857684938019842e-05,7.26835495914702e-05,8.123003978105152e-05,7.914854802778317e-05,8.064581526745279e-05,9.83149152853967e-05,9.596386533063377e-05,8.721799836432696e-05,8.544564723550072e-05,6.367774119357882e-05,4.637123462575127e-05,3.796942202254833e-05,1.1617501236211615e-05,-7.218087103721324e-06,-1.7827326334892225e-05,-2.6054822120681707e-05,-2.3244758857045046e-05,-5.082911791891287e-06,5.3487497558740096e-06,1.205832924750753e-05,1.7902334813286168e-05,1.7535958940475096e-05,2.9642157187428707e-05,3.358829866882521e-05,2.3655498432422997e-05,2.3316927952462072e-05,3.2596298420189985e-05,3.721433956646786e-05,3.0473026292131478e-05,1.5759628889576828e-05,1.0864990176139293e-06,-1.0896569394367312e-05,-1.2377148740728571e-05,-4.17022085633874e-06,-1.1523641147742051e-05,-2.982938327507767e-05,-4.0778238685244775e-05,-4.2923205945831354e-05,-3.7242523130966877e-05,-3.156221353318876e-05,-3.550470796080087e-05,-3.838158574945636e-05,-3.852621708028873e-05,-2.8297633128502713e-05,-1.1270781288050686e-05,9.557099200159287e-06,1.7974499774995867e-05,1.6735059562315663e-05,1.6952305127756754e-05,3.0940884481802474e-05,4.7518721147931276e-05,5.216051606355559e-05,4.28508541197954e-05,2.6367512275640934e-05,1.4737121805800194e-05,-5.555663782740693e-06,-2.8726530410312215e-05,-4.7086583386065695e-05,-6.623158850630922e-05,-8.080953948959647e-05,-8.125307900262043e-05,-7.24146560296037e-05,-6.0404009878080944e-05,-3.7298073078406554e-05,-1.6510553495999274e-05,-3.114811813040486e-06,2.0857973527079233e-05,4.310250460069987e-05,5.497711102318517e-05,5.259087890132819e-05,4.158750125219436e-05,1.922835339115787e-05,1.6240923964138865e-06,-1.3240505811405354e-05,-2.0584638503819547e-05,-1.8949514843453353e-05,-1.8858787736733853e-05,-2.0726260913172636e-05,-1.01517363863261e-05,-3.3520041820934938e-06,-1.7154978076825581e-06,3.423899022250931e-06,1.904966362690178e-05,3.179792059480349e-05,3.422609188236042e-05,3.2214542226879e-05,2.9970300699243706e-05,3.515762042037724e-05,4.3118545287007065e-05,5.0778284204699163e-05,4.448686487626994e-05,2.613724000019808e-05,1.782385135920935e-07,-2.735745099751423e-05,-5.15763908181415e-05,-7.202288836182469e-05,-9.426343358805867e-05,-0.00011544008190532443,-0.0001272312671814781,-0.0001175186641806784,-9.528262240069447e-05,-6.769933500526342e-05,-3.641368530381401e-05,-2.26046619565055e-06,2.2499787402858264e-05,4.2409601532894974e-05,5.1324219809675526e-05,5.831064710368297e-05,6.64218437079928e-05,0.0011311165348025086,0.002577308410544192,0.003031874586712823,0.0025944760836029866,0.0016241751663709674,0.0005318766937990166,-0.00048303565415753493,-0.0014206012929876017,-0.002134873975624069,-0.0025472161204567584,-0.002592601314725699,-0.0023022372556057475,-0.0019280487989775128,-0.0013759434954395587,-0.0007432330431545444,-0.00018494762097420682,-1.1827572793701595e-05,0.0005284017996736622,0.0009359760848492798,0.0011494567687955885,0.0011738819122184387,0.0007597204359854588,0.00034110406487998187,3.7982874428928676e-05,-0.00016679757471925019,-0.0002799533563782137,-0.0006896683643299196,-0.0007063595074340807,-0.0005657115678783398,-0.00032325209887430826,-4.0643053664313285e-05,-7.927206970583703e-05,-8.137928265001354e-05,2.8130790226367608e-05,0.00019305615793455328,0.0003612427322602074,0.0003124407727095535,0.0001856097607026661,0.0001275915873850696,0.00012849379559899546,0.00017158766478040592,0.00013010056171199367,1.1848512617174909e-05,-3.439639727341909e-05,-1.571259937101882e-05,5.1967235476458786e-05,9.31550348999125e-05,5.434047126398231e-05,4.63049542295533e-05,7.251812304798326e-05,0.00012596811515368223,0.0001729437183724366,0.00015606666677129045,0.00012575995544220335,0.00010597701983558522,9.548678591590614e-05,8.71041173462575e-05,4.791594949432986e-05,1.888829237928714e-07,-2.9505723179030792e-05,-4.560093900857066e-05,-4.886106198691981e-05,-4.7069536019773624e-05,-6.481689137901621e-05,-6.936743979720922e-05,-5.428200999391149e-05,-2.463633584549414e-05,1.834281438472515e-05,1.6216952672567277e-05,1.0954021592081742e-05,6.014044367264955e-06,3.648662070708757e-06,8.099577412814529e-06,-4.500930806017975e-06,-1.51483205452269e-05,-1.9182621017037487e-05,-2.3681830612962446e-05,-7.228098034643456e-06,4.617805475854722e-06,1.0199448671194307e-08,-7.621342496236507e-06,-1.2910106641531236e-05,-9.66489151005894e-06,1.7062940368180714e-05,4.675623531486111e-05,7.586433943855252e-05,8.943015797903028e-05,8.773067668430061e-05,8.163568404522013e-05,4.830007054704087e-05,1.572183752824734e-05,-2.106311482286663e-05,-4.795569462641844e-05,-4.745376278888323e-05,-6.184106733827041e-05,-6.807616041167081e-05,-7.730090590343286e-05,-8.64839704430358e-05,-6.401511614333944e-05,-4.9330247118919807e-05,-2.3458935454167163e-05,4.6723690855233596e-06,3.483534912692504e-05,7.46727418995076e-05,8.205268574634161e-05,7.160860900718868e-05,6.0940817072016345e-05,4.923513420367028e-05,4.505079665663818e-05,3.0493558107877185e-05,1.3765723610952875e-05,6.341714703764018e-06,8.20859085156231e-06,1.527006392186641e-05,1.0309352101347889e-05,1.9289661315523973e-06,5.411755888478308e-07,3.386659966870812e-06,1.2309541154284919e-05,6.823092707378794e-06,-2.0615944578736085e-05,-4.2438536414495094e-05,-5.243677492266498e-05,-5.666425516654978e-05,-6.217200862611115e-05,-6.798899768006217e-05,-6.747142353537334e-05,-4.562028318878222e-05,-2.5805667851460045e-05,5.125287319925717e-06,2.7693774617671134e-05,3.6431808344910184e-05,3.491584936071142e-05,3.173077216177383e-05,2.5224041325664425e-05,7.541316959713142e-06,-4.768140014980296e-06,-1.5299622910338098e-05,-1.6454889729868914e-05,1.730321003050483e-07,8.106376966490245e-06,1.0392700273486011e-05,5.103379028989277e-06,4.5562564939089455e-06,2.0225639120226767e-05,3.1083617803330345e-05,4.028253167559181e-05,3.602143640351105e-05,1.109268026900372e-05,-1.1472175814012874e-05,-2.9850182024760113e-05,-5.0573307645750744e-05,-6.955988296904664e-05,-8.301447619327822e-05,-7.852120039339174e-05,-5.7021318269557175e-05,-3.718002807551447e-05,-1.698955561541094e-05,2.687362699736113e-06,2.2476834120963875e-05,5.1635935407762034e-05,6.86441696595353e-05,7.585127258786958e-05,8.219219943184519e-05,8.316502899549854e-05,7.846824765254308e-05,6.42794894139236e-05,3.785871247520066e-05,1.3008202829567474e-05,-1.5585191428856203e-06,-2.0712636498639572e-05,-4.094029342041421e-05,-5.831050397684811e-05,-5.960963604095421e-05,-5.342411730468878e-05,-4.1834781890968535e-05,-4.2828960041324417e-05,-3.208814719506093e-05,-1.5768221173755297e-05,7.398587102670023e-06,4.0292337346945205e-05,6.551428494000792e-05,7.679957690113723e-05,7.883929124230249e-05,7.546228180084914e-05,7.711143978378244e-05,6.558223041877404e-05,3.616188622080115e-05,9.681982795847785e-06,-1.8178801425421548e-05,-3.6688373433363976e-05,-4.943506554725673e-05,-5.237792086005467e-05,-5.538465762100556e-05,-6.756923456185909e-05,-7.916361429335867e-05,-8.224494536205988e-05,-7.448660901457979e-05,-6.0026490229256526e-05,-4.4304721601184345e-05,-3.084899676716684e-05,-2.0522819643793678e-05,-5.198116751774699e-06,2.6212924187886792e-05,5.9603364129617794e-05,7.485595665913785e-05,8.627835521809741e-05,9.721220349804214e-05,0.0001137040821666751,0.00012137770972474854,0.00010664602395373984,7.982861911283917e-05,4.474282265616838e-05,5.169799069922013e-06,-2.9685822290853675e-05,-5.9535025790761696e-05,-9.071147088616833e-05,-0.00011862460515078489,-0.00014550468977756335,-0.0001684586122754301,-0.00017067972875481493,-0.00014821441561745476,-0.00012258460704356628,-9.417735534431357e-05,-6.23571497052258e-05,-3.0949772571873207e-05,1.535695172733747e-05,5.88701276380687e-05,9.048945747477579e-05,0.00011085472768953426,0.00012189310628957906,0.00012748581201214504,0.00014591407061652307,0.00014995839638396686,0.00013649184778784866,0.0001058424971597986,6.796285073124375e-05,4.2210001079610126e-05,2.8234672436422707e-05,-1.3494716572256152e-06,-3.8284884776623435e-05,-7.53167886178969e-05,-0.00011308591408145524,-0.00013687926442016156,-0.00015071016237695785,-0.00015559196597120044,-0.00014134854098631671,-0.00010744409325954783,-6.627602652662961e-05,-2.2338857958712713e-05,2.4793146207001978e-05,6.909561473169074e-05,9.699649686478253e-05,0.00010621884866234975,0.00010762072973497362,0.00011021466547737881,0.00010345858326576988,9.012548171857299e-05,6.403761264287931e-05,2.7811284551664778e-05,-9.049760654821515e-06,-3.7473697479906965e-05,-5.733595589201864e-05,-6.424117844654007e-05,-6.783657235128321e-05,-7.131445430373152e-05,-6.664439409533614e-05,-5.4501646767160735e-05,-4.7210753680642104e-05,-4.71791774290322e-05,-4.828226034998677e-05,-4.494875351085509e-05,-2.622768708375463e-05,-8.7564419690811e-06,-3.910722071521874e-06,-5.558214348875407e-06,-1.023359533416564e-05,-1.7277518227190402e-05,-1.3136509318536811e-05,-1.1690243676179396e-05,-1.1456701922981285e-05,-9.045605230735033e-06,2.597888966429915e-06,3.203896482638465e-05,6.535437082733745e-05,7.523908584249667e-05,7.79003587856759e-05,7.709634637028344e-05,6.768564014886996e-05,5.696001402392391e-05,4.495779557781024e-05,2.391643775742767e-05,1.4338449517010136e-06,-2.2204900603318907e-05,-4.3699502044210664e-05,-5.4809450930973554e-05,-6.486566855485005e-05,-7.646568363418752e-05,-7.378757114651263e-05,-5.570053365681884e-05,-3.0645970660185484e-05,3.091071903669026e-06,2.6321597633184376e-05,3.813664193747559e-05,4.073994673930415e-05,4.324853512885912e-05,3.721439137659136e-05,2.818662964129184e-05,2.1180616044137764e-05,2.643508536360358e-05,3.698150309296546e-05,3.915327459829015e-05,3.053970220299832e-05,1.6881444426142893e-05,7.994202733727121e-06,-2.0343296220603607e-06,-1.7347781416671087e-05,-3.214889453986875e-05,-3.35808770170811e-05,-2.4784631667697114e-05,-1.4092828836600168e-05,-1.9880445073719788e-05,-3.282723996388406e-05,-4.0044561572853125e-05,-3.5529630713602044e-05,-2.9674035251400788e-05,-2.2148527568264335e-05,-9.621339386070374e-06,4.91710797005255e-06,2.8493115061504164e-05,5.8486340480633354e-05,8.446586609356029e-05,0.0001020673786496693,0.0001083374981447782,0.00010963292882140733,0.0001026180385705552,8.874398377585597e-05,6.484101100382804e-05,2.9320072165217497e-05,-9.643171902302665e-06,-4.361631500877822e-05,-6.0906601813994095e-05,-7.854615178877863e-05,-0.00010000273851396147,-0.00011625542494366233,-0.00012117877006897019,-0.00011771025101052375,-0.00010614870667183833,-9.521243869477496e-05,-8.744941388246667e-05,-7.502345628308383e-05,-6.110342526694552e-05,-4.463437588673442e-05,-2.460251527948604e-05,9.8863759762858e-07,2.5714424348870067e-05,4.433437929116394e-05,4.876106844303041e-05,5.1375401619845205e-05,4.4056994575559246e-05,3.456021443691121e-05,1.5399789554024994e-05,-6.3041037695376575e-06,-2.367046236560771e-05,-2.5330604911636247e-05,-1.635537462154834e-05,-2.17530077980899e-06,3.130176172801299e-06,-1.697952670815493e-06,5.606692699534556e-06,2.9988819515103736e-05,4.571939180740168e-05,3.8557078404175435e-05,2.979369368922067e-05,3.509939513436406e-05,4.754970406546411e-05,5.128101105445697e-05,4.603250282185519e-05,3.617573241312559e-05,2.888251413252663e-05,3.5082188243895685e-05,4.346193735368709e-05,4.383357656378195e-05,3.729626449922449e-05,3.072688662635949e-05,2.336176083734581e-05,1.678133527599262e-05,5.40945678246139e-06,-2.2976909678529435e-06,-5.469893654288557e-06,-8.768893504979902e-06,-8.096567024541937e-06,-6.371700706610168e-06,-8.80995421944346e-06,-4.987708634882373e-06,-2.5462281488325554e-06,1.1200858702385796e-06,3.9532142506396246e-06,-2.754874737694882e-06,-2.1499011445124408e-05,-4.3885188932440016e-05,-6.037304484769815e-05,-6.540779850800853e-05,-7.052560809346521e-05,-6.725145124746391e-05,-6.732837355063567e-05,-5.428805483812308e-05,-3.738300001954902e-05,-2.2318508627993102e-05,-9.393017827605975e-06,6.166371592490316e-06,2.4506907622855042e-05,4.1716340344445184e-05,4.957847580081877e-05,5.157140088455167e-05,6.121093480812567e-05,6.448657365169154e-05,5.66268787005203e-05,4.348171181463162e-05,2.431715822237245e-05,3.240883873014215e-07,-1.3817927562853252e-05,-1.9870663623746387e-05,-1.7383396342493096e-05,-7.777517604413417e-06,-9.185220439131542e-08,3.3153495369192158e-06,6.471390848732389e-06,3.6945845752440304e-06,-4.006864564300307e-06,-5.721898785773711e-06,-1.2079927947511307e-05,-1.3587781097575016e-05,-1.2281657799819909e-05,-1.1165094958141833e-05,-1.0979731027174508e-05,-1.483034020166288e-05,-1.2031524275924258e-05,-6.322380507893868e-06,-9.25254763909929e-06,-9.169703397190958e-06,2.7946052744449093e-06,2.4301170196869713e-05,3.4819969011862124e-05,3.707640044402683e-05,3.651729434935619e-05,4.2765906839315555e-05,5.13894065312186e-05,5.283651107765093e-05,4.022489349340362e-05,3.6626331092808834e-05,3.63809644688471e-05,3.3217881478398314e-05,1.546612056755409e-05,-7.2756812390187125e-06,-2.887430954708762e-05,-4.890990396507118e-05,-6.357035313177144e-05,-7.705899054339159e-05,-9.529055306093118e-05,-9.327194167521061e-05,-7.335898100614785e-05,-4.122089683199394e-05,-1.336508349044199e-05,7.042406088648234e-06,2.5147682682326304e-05,5.280490538585764e-05,7.115370516820161e-05,7.177262255575879e-05,5.450580899655666e-05,3.0724141594798724e-05,6.591673206010781e-06,-1.2711517116998528e-05,-2.5912475729807424e-05,-3.740873033965481e-05,-4.896135300281804e-05,-6.319480922531617e-05,-7.671847193381721e-05,-7.386921868783852e-05,-5.929711518354569e-05,-4.8124494721387116e-05,-3.570705602572482e-05,-2.158819294097212e-05,-8.96236368309056e-06,9.178674734288813e-06,2.972544499144031e-05,4.438506850183967e-05,5.70188586299402e-05,5.8483249915493284e-05,5.271132464775831e-05,5.0065015963757195e-05,4.012503619363169e-05,2.5493501545723038e-05,1.5636020893193575e-05,3.627877354613055e-06,-4.488348694153364e-06,-3.9000229649798865e-06,-5.381975636248579e-06,-5.823530271800657e-06,-2.2850182144855743e-06,8.031647623344871e-07,1.3022462437880264e-05,2.9419951712843862e-05,3.654057095661246e-05,3.947898173686605e-05,4.3153260332474956e-05,3.9725312677495064e-05,2.7132829675616418e-05,6.041149470027588e-06,-1.6581982421272418e-05,-2.7831224079209803e-05,-3.672890897784449e-05,-4.2659295010685246e-05,-4.380321089328117e-05,-3.995237548395551e-05,-3.396040592098108e-05,-2.1623098419628793e-05,-3.799618614723612e-06,1.3334958461061492e-05,3.269540338172171e-05,4.614399092822921e-05,4.931035541171861e-05,4.8387441465051286e-05,4.534212374272937e-05,4.2325732686080846e-05,3.8869210914676285e-05,3.384153265974452e-05,2.1430389372267437e-05,3.08427410244955e-06,-1.5654529692687146e-05,-1.9437849341483904e-05,-2.2078507317308657e-05,-2.8105928892815035e-05,-4.503629139370806e-05,-6.155970395561794e-05,-7.544609198553979e-05,-8.913542156686894e-05,-0.00011803045626192389,-0.00014897133651561317,-0.00018488738812396193,-0.00023467200497908846,-0.000300656452848313,-0.00036422843071109843,-0.00042112302582895624,-0.00046883992673582613,-0.0005131146488190065,-0.000579868925176417,-0.0006496759928692025,-0.0007151256168731412,-0.0007741881456550994,-0.0007986975106852662,-0.0007669929581198601,-0.0006937494534162585,-0.0006048150428508095,-0.0005302995717096411,-0.00036090532132893196
    ]

    ACT32_RIGHT_HAND_POS = [
        0.0002746941200667411,0.0002806663545605928,0.00028689015270196293,0.0002932385197925989,0.000299058363772889,0.00030592338412669247,0.0003139394263400303,0.00032174723514591174,0.0003305779300730846,0.00034082308557099465,0.0003526924935359213,0.00036446950154236176,0.0003775865975558393,0.00038988846344073557,0.0004000364438961112,0.0004111754005982299,0.00042338108044390573,0.0004364500843826503,0.00045018418901575575,0.00046438329903504034,0.0004782033724130814,0.0004932074614975101,0.0005104490052791448,0.0005279194760857824,0.0005451581415764694,0.0005642272533091438,0.0005818046993922827,0.0005982011835911037,0.0006141037726552971,0.0006284624763823347,0.0006414645176070346,0.0006538881021961126,0.0006636014840698335,0.0006715594689037377,0.0006781127200279225,0.000683847816837192,0.0006889625084868185,0.0006934692063818968,0.0006978107933531736,0.0006988577838899756,0.0006989424533824232,0.0006998603755721792,0.0006980865755869112,0.0006942280333987117,0.0006916044684558688,0.0006900569549719756,0.0006880323571024612,0.0006841957184060038,0.0006816377379311233,0.0006785896594083983,0.0006750376555087552,0.0006720799043302444,0.0006684952917160552,0.0006637285492876824,0.000657363136272382,0.0006515179495919111,0.0006445385845728693,0.0006355022391722162,0.0006246483240692694,0.0006141981582502013,0.0006030485345396276,0.0005898426876831277,0.000576409424407391,0.0005626024110040989,0.0005478872552162012,0.0005310256820192034,0.0005136597548194001,0.0004961476673763706,0.00048051183358856263,0.0004642739988653271,0.00044747833401875326,0.00043061629547630644,0.0004128775729558573,0.0003945246638681966,0.0003760023350768639,0.00035534702544994776,0.0003331503021556474,0.0003117666320741083,0.00028940922429165585,0.0002678637110379283,0.0002475951388574565,0.00022879191352807548,0.00020952801990423855,0.00019241089200962966,0.00017649514909675524,0.00016027000488089905,0.00014415910643790484,0.00012741613084760932,0.00011150126443663293,9.52310597857801e-05,7.930863707774559e-05,6.250745480934851e-05,4.591961869160416e-05,2.8586778771683223e-05,1.0949123540781024e-05,-7.479892396900472e-06,-2.6778933701493614e-05,-4.2852286367888836e-05,-5.696859967796803e-05,-7.098504732399757e-05,-8.302424132560999e-05,-9.33619963010177e-05,-0.00010208439292789914,-0.00010872826224602793,-0.00011537330940577412,-0.0001204921992718881,-0.0001244931581353706,-0.000127411857928525,-0.0001306615353973507,-0.0001329102423469176,-0.00013486684328160249,-0.00013556603224291995,-0.00013703503703745696,-0.00013813901615794292,-0.00013930160678633582,-0.00014178527192716992,-0.00014443217703997174,-0.00014634392240558836,-0.00014903270231093102,-0.00015299731141249856,-0.0001556034765405987,-0.00015816346975377468,-0.00016184228534222203,-0.00016531068198027965,-0.00016998646571164506,-0.00017505575065213165,-0.00017869642229688138,-0.00018283028812924324,-0.0001873422284612865,-0.00019315611012521528,-0.00019986759943645084,-0.00020804930369405574,-0.00021440403901650567,-0.00022020789143440225,-0.00022613777595425255,-0.00023064625024326387,-0.00023290191638131793,-0.00023691226635846278,-0.00024229154567629996,-0.00024655310467516827,-0.0002507230543630599,-0.0002551706689360465,-0.0002600023374063071,-0.0002640927928065784,-0.00026715945381958325,-0.0002696027224042547,-0.00027328096846639674,-0.00027580507414488273,-0.0002773300659261591,-0.00027796968592887623,-0.0002784492812641722,-0.00027844897033662457,-0.00027885614165753543,-0.0002790301954282053,-0.0002776023025504507,-0.0002756179761551323,-0.0002728495505481669,-0.0002694198828247075,-0.0002658129264590886,-0.00026232604121150297,-0.00025758879213363814,-0.0002510442701786691,-0.00024321347854577112,-0.0002340091496703185,-0.0002235442294336115,-0.00021339397978080107,-0.0002028899395152972,-0.00019109264449901266,-0.00018005256434132128,-0.00016849598437444256,-0.0001566681915300193,-0.0001455468521028608,-0.000135039000872713,-0.00012450920055828506,-0.0001129960125676189,-0.00010211126172052238,-9.393615319408008e-05,-8.672646325358287e-05,-8.04937758695526e-05,-7.483064469504084e-05,-7.079231356070306e-05,-6.919407516906889e-05,-6.7471082763446e-05,-6.716992693522543e-05,-6.846673804558044e-05,-7.124299658005328e-05,-7.335970236597038e-05,-7.634703578534625e-05,-7.986573164987943e-05,-8.333359419013097e-05,-8.721468999411818e-05,-9.201761316335459e-05,-9.85631416829242e-05,-0.00010576799154367344,-0.0001140589869956402,-0.00012369196066302061,-0.00013368569424328965,-0.0001432479412616494,-0.00015338465824863684,-0.0001635277941704548,-0.00017530067187281824,-0.00018773628171043867,-0.00020079775587804728,-0.0002146823788688277,-0.00022777267416809247,-0.00024033142300242873,-0.00025279851753617883,-0.0002650007399104776,-0.00027765107758700964,-0.0002901629476167644,-0.00030235232911385025,-0.00031478852856544926,-0.00032712587206100885,-0.0003383367993000024,-0.000349411702171456,-0.0003606226968343482,-0.0003722543402841319,-0.0003831371063279885,-0.0003932421998727182,-0.0004026537957642915,-0.0004117342641412435,-0.00042009922733696924,-0.00042854015018795494,-0.00043715161387593396,-0.000443683657191424,-0.0004487144261709278,-0.00045373956152417355,-0.00045813956807492124,-0.0004602535954783672,-0.00046111608864117157,-0.0004601875689364031,-0.00045939722093230425,-0.0004586681663107013,-0.0004567306635484821,-0.0004543888802183574,-0.0004520531191549038,-0.0004487042561261833,-0.00044466868717024346,-0.00044037154308851535,-0.00043592374880661064,-0.0004315077556884919,-0.0004266262205595181,-0.0004205294795513748,-0.0004139082878252263,-0.00040615832346788447,-0.0003968926350514527,-0.0003874440012186173,-0.0003782370264651212,-0.0003694587630456145,-0.0003622945553845011,-0.0003562965132031953,-0.0003494757054584739,-0.0003424445195066798,-0.00033550391405629313,-0.00032673382553251957,-0.00031756597347114876,-0.0003074737175765633,-0.0002984446023983645,-0.0002909330382202899,-0.0002847039104815913,-0.0002800956089311686,-0.0002754815766421302,-0.00027216225982817677,-0.0002697435913248597,-0.0002676119642094025,-0.00026540564738894207,-0.00026181498351364235,-0.00025784361828327696,-0.0002542979614546343,-0.00025028851441861723,-0.00024501648986280876,-0.00023931166750013286,-0.00023500362533685292,-0.00023045568292435513,-0.00022490474919384073,-0.00021949587486050782,-0.00021370690236599183,-0.00020760176051048507,-0.00020141753622280147,-0.00019580896180805703,-0.00019104409603782962,-0.00018547031922278828,-0.00017918231447210575,-0.00017240073888716298,-0.00016456512516237122,-0.00015603382631219277,-0.00014696651887579444,-0.0001377132571175331,-0.00012911371476472008,-0.00012114826951615068,-0.00011307461507026394,-0.00010514960847311686,-9.781950763649428e-05,-9.022001872372044e-05,-8.306163514562951e-05,-7.623133482855119e-05,-6.86652980605072e-05,-6.179157139494246e-05,-5.5813232278467445e-05,-5.004369459581018e-05,-4.48061172082206e-05,-4.061597606511542e-05,-3.632621988077558e-05,-3.1986014495997706e-05,-2.8055843366148204e-05,-2.395176134834341e-05,-1.9849253003575995e-05,-1.548579792974801e-05,-1.1607107105859115e-05,-7.923393310190867e-06,-5.763358945021109e-06,-3.221305621006728e-06,-1.3765228667817053e-06,7.515849628813841e-07,2.110229155885002e-06,4.5913918589654995e-06,7.352621371083818e-06,9.679534639133473e-06,1.1796610825239109e-05,1.4171785038850433e-05,1.6331752370974366e-05,1.822454776914314e-05,1.9580072928647803e-05,2.1728186226989154e-05,2.421355823183112e-05,2.5404405708581347e-05,2.5488264388211056e-05,2.598608857573203e-05,2.627571366540905e-05,2.597246230491783e-05,2.4813695478678196e-05,2.2672096502175594e-05,1.977442188453441e-05,1.7139858518124934e-05,1.4840827637441621e-05,1.2122136294051935e-05,9.053313823705015e-06,6.5039591001030324e-06,3.686914704951827e-06,1.206893716489519e-06,5.795988394714778e-07,9.812560990380635e-07,2.5875500454071196e-06,5.679703736591138e-06,1.002104545784704e-05,1.3792062073813293e-05,1.7046693976504852e-05,1.9961892357946926e-05,2.3051005522618194e-05,2.5754254903268843e-05,2.6249237063974444e-05,2.7115060883682797e-05,2.965677851384641e-05,3.189387041150842e-05,3.435235704648418e-05,3.714815491556728e-05,4.0087085752923647e-05,4.3304387462213576e-05,4.720266859725866e-05,5.167513809500004e-05,5.713638955150849e-05,6.334846878206443e-05,6.891320637955254e-05,7.56147114079451e-05,8.284436647044399e-05,8.98939286557045e-05,9.639628451193311e-05,0.00010203232741051579,0.00010627314654963612,0.00010944851939967171,0.00011262998876551069,0.0001144691971646503,0.00011442779621454054,0.0001135083434684412,0.00011219001160612998,0.00010982657823788498,0.00010579085317915942,0.00010210828032900558,9.767579722391499e-05,9.349622118613482e-05,8.802413191998176e-05,8.237890563437084e-05,7.642893920740322e-05,7.067659511086358e-05,6.434050828298685e-05,5.716904090469276e-05,5.0307679246839714e-05,4.3018186781855825e-05,3.484684721682571e-05,2.6182566289554634e-05,1.678784267861843e-05,6.544176170164002e-06,-3.7765809836205915e-06,-1.394192575374485e-05,-2.3904702531919287e-05,-3.5214450565091386e-05,-4.7156892901972975e-05,-5.812156216415179e-05,-6.83788306391156e-05,-7.923127419637081e-05,-9.074326940105793e-05,-0.00010228207072634245,-0.00011267573692685656,-0.00012419210952220735,-0.00013644679476138314,-0.0001473987364320309,-0.00015720611782172366,-0.0001669938594080487,-0.0001766029876024111,-0.00018517598759759408,-0.00019245948907930003,-0.0001996673893363525,-0.00020613558475061108,-0.0002125107169160769,-0.0002178585898736739,-0.00022096732773058302,-0.00022417773910094678,-0.00022691896229890454,-0.00022865455695028356,-0.00023013813217943492,-0.00023196174497896213,-0.00023271623062148922,-0.00023269105635252382,-0.0002331377388340954,-0.00023504907911503785,-0.0002358988927018388,-0.00023762400814851333,-0.0002390678193466914,-0.00023825634649754816,-0.00023692942467485388,-0.00023499659828796484,-0.00023321912572105835,-0.00023174262427730387,-0.00023036384486336184,-0.0002259914263897094,-0.0002222843215458094,-0.00021961674782760386,-0.00021489857872589657,-0.00020827128399362926,-0.00019917442449629,-0.0001889926401115275,-0.0001769102121327097,-0.00016469385053053003,-0.00015114459280035514,-0.00013604583070241102,-0.00012036740570134503,-0.00010464440851274384,-8.919044165084103e-05,-7.537334633263213e-05,-6.29698680649656e-05,-5.188077442193765e-05,-4.148786878536953e-05,-3.318972104821604e-05,-2.8662937232814908e-05,-2.479723821395227e-05,-2.1809422815032658e-05,-2.041649191286481e-05,-2.007816843968792e-05,-2.1539952750056456e-05,-2.347455552553903e-05,-2.5355341635166913e-05,-2.7891263071992627e-05,-3.106505813866445e-05,-3.444481759978339e-05,-3.6103153052509234e-05,-3.577784700078105e-05,-3.5355923965321547e-05,-3.533647116503239e-05,-3.428748010674281e-05,-3.3436415663009705e-05,-3.230512309495985e-05,-3.1936089936820084e-05,-3.329583943488611e-05,-3.427525655660929e-05,-3.7889298481944763e-05,-4.473988595223322e-05,-5.2554483155382976e-05,-5.9844426832516195e-05,-6.736092539118538e-05,-7.811376523123165e-05,-8.84995396433078e-05,-9.766333806109393e-05,-0.0001080836716933179,-0.0001181664314273097,-0.0001310318058884163,-0.00014425015317593288,-0.00015427364604200225,-0.00016306145707908505,-0.00017292714260112936,-0.00018276952417875387,-0.0001913082989489384,-0.00019824542946348292,-0.0002033425457675639,-0.00020837289895408233,-0.00021307344553264893,-0.00021956492144183492,-0.0002247437990153395,-0.0002306335561843783,-0.00023821092660726714,-0.0002473565462113706,-0.00025634143898060365,-0.0002649608369434496,-0.00027459682937714405,-0.00028332280664038824,-0.00029156940500797576,-0.0002998969753713958,-0.0003089018138429219,-0.00031623115012968786,-0.00032369593799738764,-0.00033089003227195347,-0.00033702451547164893,-0.00034377141163988373,-0.0003504842570298768,-0.00035698152034603024,-0.00036149350693035187,-0.0003644843893666626,-0.00036592317225608373,-0.0003645780615850224,-0.00036463025044424646,-0.0003643941282846022,-0.00036138344348604114,-0.000357120788218876,-0.000352144201331154,-0.00034506309257194273,-0.00033795213862542996,-0.00033022349027149776,-0.0003218014159119087,-0.00031303259413348545,-0.0003053248132851415,-0.0002977554298707906,-0.0002895981872445312,-0.00028205988170842223,-0.00027458109285968243,-0.0002664623734899084,-0.00026164998630755627,-0.0002599007537783824,-0.0002582633715993705,-0.00025730790200446436,-0.000256855028768485,-0.0002569536466633558,-0.00025749417238567874,-0.00026011909397495055,-0.00026389010289160314,-0.00026809128767130364,-0.0002714443085326594,-0.0002743686680796368,-0.000275596324745205,-0.00027697227340933223,-0.0002782921558396608,-0.0002788121911084176,-0.0002791431827098458,-0.0002789286981736872,-0.00027668129044028497,-0.0002739643646694374,-0.0002712607983523043,-0.000268533277399866,-0.00026693460231930004,-0.0002671929340877533,-0.0002669164982984098,-0.0002674278192403443,-0.0002686086562730454,-0.00026915407574521047,-0.00026929131632178355,-0.00027090913498977896,-0.00027217019821536135,-0.00027199086674532314,-0.0002709591806847865,-0.00026907516805134597,-0.0002663170113782931,-0.000261945043683669,-0.00025348300492745974,-0.0002443420704658031,-0.00023580717830726317,-0.00022546116382786785,-0.00021461254144113624,-0.00020651692784486093,-0.00019808589698738018,-0.00018941510162829653,-0.00018131222218999927,-0.0001726683639975326,-0.00016432758369722006,-0.00015556698279679606,-0.00014685585119081945,-0.0001377393588464388,-0.00012840191383750303,-0.00011941362756741676,-0.00011063255884847922,-0.00010028296364375103,-9.065041271505809e-05,-7.995873651964328e-05,-6.811803671318637e-05,-5.6283228104412e-05,-4.4341773058577806e-05,-3.3082066826678115e-05,-2.109783651713443e-05,-7.27705912484574e-06,8.614163584701634e-06,2.4288678566897654e-05,4.019167446572639e-05,5.59076534879424e-05,7.124317575899196e-05,8.645884249773652e-05,9.934795386913117e-05,0.0001105508936089053,0.00012122388105390322,0.0001319706901443068,0.00014345514177894317,0.00015622864125229314,0.00016717599299750593,0.00017772096870598319,0.00018900148269190422,0.00019899810473146564,0.00020747791617494834,0.0002161768638602767,0.00022436516547763924,0.0002342151520740547,0.00024551064139694136,0.000256464954406242,0.0002699203044670163,0.00028351296206479814,0.0002966019105956162,0.0003097553743100108,0.0003229980114183951,0.0003341271238518231,0.00034443954663536234,0.00035405372365137516,0.0003617112736490181,0.0003680982224461293,0.0003719919715732336,0.00037533110692362063,0.0003771806838423623,0.0003786026765499238,0.0003786902937046684,0.0003764640907471129,0.00037464891168781633,0.0003722986601469407,0.00036950994006112683,0.00036713749348758545,0.00036518343570109173,0.00036444469495611637,0.00036374936340914736,0.00036283130615623327,0.00036252268953320717,0.00036367616602797436,0.00036500406329367926,0.00036653488607243267,0.0003687795491217286,0.0003717941845683578,0.00037630173937562664,0.0003803149384468379,0.0003831747979848789,0.00038649002378575155,0.00039015481292272494,0.0003943826599685857,0.00039854089263252434,0.00040242054852834076,0.000404579530966576,0.0004059265635330098,0.00040558606957286016,0.000403953390615144,0.0004001666322415327,0.000396402861023312,0.00039172921021446644,0.00038579724246955383,0.00037916303414443735,0.00037252139240579753,0.0003663708152298228,0.0003611342539028059,0.00035610285250931023,0.0003510759983741225,0.00034434120573082,0.00033653291312385697,0.00032712986300341114,0.0003168251437760633,0.0003053749936191883,0.00029359374105721785,0.00028158133407891675,0.00026872394231471937,0.00025587360907549896,0.00024279815402675147,0.0002302577122121887,0.0002181234003285792,0.00020654376621524368,0.00019598617294713927,0.00018672654923491632,0.00017747318132136562,0.00016758302528845862,0.00015915689090608437,0.00015195632281785285,0.00014437621938193253,0.00013638413878728613,0.00012889809700918593,0.00012141739363001501,0.00011554550979006389,0.00011087366083690257,0.00010616384149551855,0.0001011187691118192,9.617756486628707e-05,9.17699129876806e-05,8.875837368392997e-05,8.666024047368005e-05,8.656170268444164e-05,8.765948186819995e-05,8.887239987756891e-05,9.011665354844997e-05,9.029577618071085e-05,8.946258464431283e-05,8.853668235036013e-05,8.77387008551479e-05,8.656755885403184e-05,8.45109327119146e-05,8.198365374595222e-05,8.040086519469615e-05,8.036422969984124e-05,8.030043314676093e-05,8.122537221939704e-05,8.320681359387602e-05,8.621624655961198e-05,8.989856013445427e-05,9.53547908812353e-05,0.00010178350861643704,0.00010883651232406566,0.00011660923721767382,0.0001248016463045036,0.00013281290929834834,0.00014180648584200545,0.00015165398236733068,0.0001612985503760697,0.00017051819349808876,0.00017958647090108706,0.00018891067970313373,0.00019797812728238517
    ]

    ACT32_LEFT_HAND_POS = [
        0.00034031108530035797,0.00034945613421226456,0.0003588384922420407,0.0003691233170840799,0.0003804027647670369,0.00039372569668160877,0.00040843118444524104,0.0004238653392858636,0.0004404492280640941,0.00045648782966231286,0.00047391093045139136,0.0004927013197156065,0.0005113876686535375,0.0005300414873939109,0.0005487457518238375,0.0005677790891930032,0.0005869090748651455,0.0006053762667978826,0.0006231070256121533,0.0006406558445834601,0.0006592102569446461,0.0006762836410665196,0.0006922303856266063,0.0007067170301224647,0.000721447580942873,0.0007354698140802787,0.0007488287494092028,0.0007621631025472327,0.000775572975943013,0.0007904501020124418,0.0008038188692264064,0.0008186354081618658,0.0008335832303587708,0.0008470189290065391,0.0008600336170253678,0.0008736776160301896,0.0008863569218120756,0.0008976697647001372,0.0009088364155850917,0.0009200896658065918,0.0009316513705719731,0.0009426579994834713,0.000952468717385456,0.0009637055902288627,0.0009747710227251002,0.0009851487139239904,0.0009937188179183897,0.0010028905854074972,0.0010108807316749828,0.0010169925791507458,0.0010230295531856066,0.001028527346732328,0.0010311349658855952,0.001033730104799251,0.0010351577152236853,0.0010357961144737335,0.00103489795439117,0.0010323510923894747,0.0010284256333004007,0.0010227881764677685,0.0010152510943937131,0.00100531160222178,0.0009941335932680492,0.0009815742786530928,0.0009687830469676989,0.00095518049803474,0.0009409197419122527,0.0009274982511451825,0.0009134945635261223,0.0008984395293683946,0.000884682618328326,0.0008696006599636913,0.000853380450891662,0.0008361479482355004,0.0008208561808171907,0.0008052743591548897,0.0007901831207664237,0.000776212997033472,0.0007610601001936493,0.0007465060071846987,0.000732257043448893,0.0007165522300629528,0.0007014145051722735,0.0006873361224702066,0.0006731809942385081,0.0006559693089664991,0.0006365493965677951,0.0006182467216124251,0.0005988580814407068,0.0005785749281647831,0.0005586389656579357,0.000537995264532212,0.0005176338364943481,0.0004978550879820956,0.0004783330466061251,0.00045767030306162715,0.00043870933189576474,0.0004212607088446118,0.0004027568358262361,0.00038451503462702324,0.0003664326076552664,0.0003509796388414816,0.00033603887352638505,0.00032019317316089695,0.0003043660898255852,0.00028982523351576034,0.00027582394429737214,0.0002629697965132102,0.0002511087895159024,0.0002407622252571778,0.00023214878314960493,0.00022371375764064882,0.00021560297946157715,0.00020797021971637883,0.00020250425709261035,0.00019739768551475055,0.00019243798559845075,0.00018816630802553618,0.00018390145565509158,0.00017816592612770947,0.00017192908182325743,0.0001640194156683263,0.00015495649423198014,0.00014546380713325238,0.00013665861203928016,0.00012662315102963693,0.00011537448614377546,0.0001025334151242565,8.859112013356143e-05,7.346132782755226e-05,5.7859594061308103e-05,4.186561518677793e-05,2.5611048905573584e-05,9.960589842590841e-06,-5.0312496542640466e-06,-2.0907384714910843e-05,-3.694657947560653e-05,-5.2727296085797595e-05,-6.8622756342113e-05,-8.577404069444012e-05,-0.00010194456067989594,-0.00011743287082397922,-0.00013217222590853177,-0.00014612463714230007,-0.00015922548435533734,-0.00017139604433167013,-0.00018359167041798207,-0.00019441546660316808,-0.00020548802766553166,-0.00021329277569468103,-0.00022178475599941098,-0.00022914597908545872,-0.0002368158217600632,-0.00024379498345175504,-0.00025059428825059763,-0.0002560151957609825,-0.000259284362618419,-0.00026277048144686654,-0.0002662480926016551,-0.00026839749399486945,-0.00027278481848304906,-0.0002769044533504267,-0.0002821450079362008,-0.0002867109218967814,-0.0002928439249982213,-0.0003004932595871982,-0.00030709948048481797,-0.0003140394744915746,-0.00032224836519607707,-0.0003295770291443406,-0.0003344642011237876,-0.0003395658054514887,-0.00034431852174838574,-0.00034616092313683247,-0.0003472259414138905,-0.00034766220418045254,-0.0003485196530724408,-0.0003502800586845398,-0.00035289170715553387,-0.00035565132701332003,-0.0003602200038193507,-0.0003665340295848638,-0.00037272034271403363,-0.0003791466435121096,-0.0003864583706358953,-0.0003939669307512104,-0.00040017461932153253,-0.000407337949466553,-0.0004156840863255823,-0.00042562794085631764,-0.0004361648150651858,-0.00044681432212180234,-0.00045814982492330846,-0.0004710434537324503,-0.0004848670327948711,-0.0004995019104603633,-0.0005147058668375492,-0.0005309058829999,-0.0005466448136744467,-0.0005636006752585003,-0.0005814645258914048,-0.0006007687817616811,-0.0006207407829620237,-0.0006412949470481229,-0.0006618014932887771,-0.0006822223546331971,-0.0007021016800228152,-0.0007207775261456141,-0.0007388025626129394,-0.0007570610711294247,-0.0007749053809718355,-0.0007910102784259397,-0.0008064698390317249,-0.0008209550764726904,-0.0008339872438299709,-0.0008464631143654185,-0.0008564454659667795,-0.0008647798246509174,-0.0008725259174909053,-0.0008784593042088015,-0.0008826288944287271,-0.000884747930640503,-0.0008855588025579851,-0.0008856752201467331,-0.0008864055240723829,-0.0008869797147671715,-0.0008868840191814921,-0.0008868739532972603,-0.0008837600436950585,-0.0008798395706500307,-0.0008756924610206639,-0.0008704685148259508,-0.0008627390965589198,-0.0008552169567176805,-0.0008477484826615044,-0.0008371938635620678,-0.0008262366880347141,-0.0008147208791920652,-0.0008026301179912143,-0.0007890381141728737,-0.0007752001311358618,-0.000760330426412917,-0.000745756099883416,-0.000730480475878721,-0.0007151180644391729,-0.0006984305240697666,-0.0006818529850393741,-0.0006639146316093214,-0.0006469719788496339,-0.0006303139916853919,-0.000613514666925508,-0.0005961248245695303,-0.0005789522974619313,-0.0005631563290726114,-0.0005471598345908765,-0.0005310212123742602,-0.0005155644476746367,-0.0005011658020615976,-0.0004860196036756839,-0.0004705771829595906,-0.00045523659264223615,-0.0004395968607198951,-0.00042296061734370106,-0.0004080217648984584,-0.00039406912273514115,-0.0003812146828173356,-0.0003676638464760851,-0.00035391121025401184,-0.00033996307601403316,-0.0003259350380766831,-0.0003121997281722842,-0.00029797641680730295,-0.000285047550864388,-0.0002730321409469077,-0.00026030090258239704,-0.00024724557589641795,-0.00023464609796816565,-0.00022388480062416507,-0.00021289884234995858,-0.00020326595043314093,-0.000193939525407891,-0.0001855578454307165,-0.00017716685724909363,-0.00016997294121335696,-0.00016346227018502695,-0.0001569644706566126,-0.0001511415563966229,-0.0001459556369005849,-0.00014146795995849596,-0.00013698419691300962,-0.000133080053296398,-0.00012920452494758633,-0.00012478742989373973,-0.00012155293547334238,-0.00011895570944488611,-0.00011616770458435133,-0.00011240365561834842,-0.00010791641111866677,-0.0001027956231844179,-9.799810081261114e-05,-9.242130906570482e-05,-8.589080533452311e-05,-7.89041643597848e-05,-7.247572261853192e-05,-6.621029120557875e-05,-5.9407356059671936e-05,-5.239488553115653e-05,-4.5675399730222215e-05,-4.0258162961692026e-05,-3.5534145425988745e-05,-2.9579483895197725e-05,-2.3249535528419052e-05,-1.6513615397399397e-05,-9.415533574366565e-06,-2.4016082448172964e-06,4.822149465302579e-06,1.2684572645462676e-05,2.1157340394160562e-05,2.9453569821109826e-05,3.622450736659685e-05,4.1226291850700286e-05,4.505560809802063e-05,4.857619589499672e-05,5.203130567376634e-05,5.527759399971151e-05,5.757759378057176e-05,5.813641245201409e-05,5.854535044678464e-05,5.846664025149629e-05,5.697841766928368e-05,5.394720142809788e-05,5.0180017259617195e-05,4.610652831626801e-05,4.2152816702465014e-05,3.8094988877773705e-05,3.523063710409778e-05,3.252443638149596e-05,3.0093395435489246e-05,2.8654143708774343e-05,2.7843407791083027e-05,2.7108411543102916e-05,2.618347520514618e-05,2.535264568099505e-05,2.429121030175267e-05,2.2514495198118166e-05,2.106029738590233e-05,2.026820460261166e-05,1.9961972620272988e-05,2.0323396838395707e-05,2.1385713946329793e-05,2.2006846591802486e-05,2.362769755378856e-05,2.635108939216291e-05,2.826043954385645e-05,2.9764260804649292e-05,3.119304508943696e-05,3.2752215865970885e-05,3.3988762670366796e-05,3.41480399509623e-05,3.4742432991710934e-05,3.5006930904896255e-05,3.423971906981235e-05,3.2231414834332064e-05,2.924454983427538e-05,2.5929096297001147e-05,2.262262554572424e-05,1.814308500494586e-05,1.2338646588555329e-05,6.460725205308152e-06,8.587291116610767e-07,-5.055394421534625e-06,-1.1510201789779759e-05,-1.6784295114758488e-05,-2.089553380225761e-05,-2.420547130234911e-05,-2.5840437947628747e-05,-2.749382313786702e-05,-2.9325339873221914e-05,-3.108728000104034e-05,-3.064646089707026e-05,-2.8549807573944607e-05,-2.7483201055821996e-05,-2.6192495856056552e-05,-2.399811386158658e-05,-2.209465013353305e-05,-2.1293026479286877e-05,-2.0912886948515818e-05,-2.2298521509390772e-05,-2.5334455397687965e-05,-2.9760174549272168e-05,-3.6053306068374226e-05,-4.4854655263877956e-05,-5.580685434683288e-05,-6.755944909480711e-05,-8.033130755043717e-05,-9.474665900458005e-05,-0.00011061341415102904,-0.0001259664687385735,-0.00014223567043798613,-0.0001570607306602611,-0.00017090303358359549,-0.00018441722096171643,-0.00019582268175911434,-0.0002042903225119308,-0.00021200823853090777,-0.00021905362898103263,-0.00022357748883147487,-0.0002267330157360441,-0.00022970434208764246,-0.00023379343599476984,-0.00023697995379145674,-0.00024111299743476456,-0.0002449614810984503,-0.000250562173925043,-0.00025725604371833806,-0.0002650376339608808,-0.00027360170569229995,-0.0002829004889070794,-0.0002929347582055466,-0.0003030926346650079,-0.00031316437370367423,-0.0003216552964955651,-0.00032919554228096276,-0.0003359665486649597,-0.0003408343408320207,-0.00034458326268770755,-0.0003475245265396108,-0.00035000841220807,-0.0003532839855341341,-0.00035695487843081755,-0.00036028181015413876,-0.0003640654623931123,-0.0003701538106925582,-0.00037668702096653917,-0.0003832656621510977,-0.0003910317449605973,-0.00040102551438208505,-0.0004116276943399958,-0.0004211790675715411,-0.0004303134809553607,-0.0004393617930561348,-0.0004488641022970099,-0.00045857801169223584,-0.00046809550217587805,-0.00047697092353382766,-0.0004840441817884835,-0.0004920233937412863,-0.0005000254702698599,-0.0005073916373659012,-0.0005178261310530911,-0.0005302924974150189,-0.0005422720353380706,-0.0005548954931263438,-0.0005699172344600089,-0.0005851919577535943,-0.0005995433075029397,-0.0006136141327771717,-0.0006278017711601878,-0.00063985755231996,-0.0006499565115832681,-0.0006575224945998902,-0.00066223051952368,-0.0006653661188487111,-0.000666653512348247,-0.0006645544995137761,-0.0006607849146010757,-0.0006549415738424649,-0.0006477695071760658,-0.0006383765150878755,-0.0006285714805481828,-0.0006182867344681132,-0.0006061307072795406,-0.0005930150848535015,-0.0005800149808057989,-0.000567096851195172,-0.0005530955830862003,-0.0005379785552711768,-0.0005242722523345573,-0.0005103652947445951,-0.0004969653206653347,-0.00048417367522149527,-0.00047057683903021233,-0.0004576233634553466,-0.00044504611673495546,-0.00043300015634041214,-0.00042093160193363606,-0.0004100633964414071,-0.0003997058878231331,-0.0003906206505808728,-0.0003832848719437939,-0.0003777392909436532,-0.0003714394267568375,-0.00036418006498061976,-0.00035762421525040557,-0.00035295026307278625,-0.0003487382666166293,-0.0003430375863011464,-0.00033755017615785945,-0.0003327099289261828,-0.00032783331920369145,-0.00032158908205144046,-0.00031476887813933386,-0.0003093478851426847,-0.0003051917020210754,-0.0003009262229777202,-0.00029669166426089214,-0.00029288481045913246,-0.000290318358314524,-0.00028684372891931613,-0.0002838278731884003,-0.0002802988940190329,-0.0002788047090781326,-0.0002772832761251554,-0.0002748609704337291,-0.00027279709514742956,-0.000270666951249343,-0.00026605055467502037,-0.0002604599965631368,-0.0002503919178555845,-0.0002418728641432436,-0.0002337909660952059,-0.00022621321805392662,-0.0002190874123812975,-0.00021274720278874012,-0.00020954581088513462,-0.00020711537078582087,-0.0002059424452134078,-0.00020557989417232936,-0.00020428731283812166,-0.0002028555896295472,-0.00020347235320233596,-0.00020488910441215797,-0.00020635988588211057,-0.00020911408731633788,-0.00021316525191648496,-0.00021648394591123634,-0.000221437027315947,-0.0002270887228205882,-0.00023186163552993747,-0.00023674322850244394,-0.00024151742500933706,-0.0002463353428055098,-0.00024987503286898514,-0.0002542471116088944,-0.00025773186936319637,-0.0002614487384518243,-0.0002646807725017847,-0.00026645945151638556,-0.0002673526316014794,-0.00026722998500271374,-0.00026684150614091854,-0.0002627232290084173,-0.00025701937797328964,-0.0002502873937472829,-0.0002432475732992872,-0.00023548178741484966,-0.00022540066417125367,-0.0002142017235090437,-0.00020227166684899745,-0.0001916929054658085,-0.00017919496633139753,-0.00016491506935548402,-0.00015097726648635818,-0.00013657052541779936,-0.0001222594033446769,-0.0001098461842033612,-9.928939188639029e-05,-8.896246218456602e-05,-7.728984443377132e-05,-6.74469947190309e-05,-5.851096149042013e-05,-4.897042857625246e-05,-3.944512300996386e-05,-3.0548564551765206e-05,-2.1002670926406635e-05,-1.0845944585591814e-05,-8.299428701571553e-07,8.614230032683476e-06,1.9268851348999116e-05,3.136204639157126e-05,4.546238008275475e-05,5.928021211248183e-05,7.334675691942216e-05,8.6843621718605e-05,9.987454186671443e-05,0.00011277339015267557,0.00012661745527632056,0.00014105773533995658,0.00015571090736459086,0.00016986195391506382,0.000183167328636048,0.00019662099087120981,0.00021059241384642613,0.0002226047335676428,0.00023411091494040672,0.0002447903974380958,0.00025574857713633105,0.00026716743849436393,0.00027815895987634147,0.0002893897251172379,0.0003008411221225421,0.0003121477171540644,0.00032197407923522357,0.0003318452043504847,0.0003427277723347607,0.00035273589914767045,0.00036181947737606114,0.0003693879555604776,0.0003746202816615183,0.0003778996874306282,0.0003807391117211787,0.00038193258115591964,0.00038269545727026306,0.0003838841040797099,0.0003857838422545874,0.00038717814388204377,0.00038795490715524413,0.0003882530913984709,0.0003896836214913924,0.00039094116389516226,0.0003926431847872357,0.0003939612964744216,0.000394678390270072,0.0003935508495877019,0.00039242373540761716,0.00039170926985836554,0.0003910976201942902,0.0003908618835398589,0.0003906679776325228,0.0003913999650563666,0.0003927525445072218,0.0003954269859147436,0.00039729049669956403,0.00039959856412425944,0.0004016558462948186,0.00040362842813966923,0.00040507463128270916,0.0004059990739263781,0.000407413437901885,0.00040845408543417947,0.00040905115771864984,0.0004096141315575003,0.00040981364356187906,0.0004085056170381539,0.0004060886254759969,0.00040544216188347543,0.00040587579301532006,0.0004072157789720141,0.0004069465574567099,0.0004073146948401504,0.0004082853382808242,0.0004089102060805139,0.0004102097135402104,0.00040921356443674464,0.00040773295804291627,0.00040556844649117983,0.00040358242555806745,0.00040206685238375616,0.0004005695156093591,0.00039880260299710355,0.0003958632560771781,0.0003921642111058066,0.0003861172608621047,0.00037902332612362743,0.00037192433443090835,0.00036595555479767487,0.0003592910481308198,0.0003521489146523305,0.00034628985311456977,0.0003407584182825465,0.0003353223547914567,0.00032919199059498776,0.0003224378441135846,0.00031501438906815727,0.0003071592345086992,0.00030078263452063686,0.0002950087028385591,0.000290751300716563,0.0002863286681621599,0.00028293075391496185,0.0002787171550964603,0.00027351703361607344,0.0002690366177803156,0.0002650694674553294,0.00026130135044354357,0.0002584962101516265,0.00025613507779940093,0.000254122650224107,0.00025325506216298974,0.00025295155784220246,0.000253661113973426,0.00025490561995189086,0.0002554431923233816,0.00025569969358263286,0.00025499465072634713,0.00025424250958529075,0.0002539439104764166,0.0002530652458601049,0.0002517363119329356,0.0002508893827134531,0.00025101051472958043,0.00025093123120895305,0.0002509843367156065,0.0002511660045304545,0.0002513185829188546,0.0002511049453272251,0.00025069425578998753,0.0002494279888912799,0.00024812987286368203,0.00024691243350803596,0.0002466300030591633,0.00024656782531633426,0.0002467547900417162,0.0002461894863794712,0.0002458096619003789,0.0002452965058596775,0.00024473780730125674,0.0002444103406816767,0.00024541610554706456,0.0002463385407651531,0.0002467510769415564,0.00024747610103169993,0.0002486161717822381,0.000250288009325553,0.0002513763735758323,0.00025304394986101737,0.0002547302451419567,0.00025703610234194756,0.0002599931360801027,0.00026320955877139567,0.00026632249550088147,0.00026931990847597567
    ]
    
    ACT32_RIGHT_HAND_VEL = [
        0.00023276822910941122,-0.00017030112367564412,-0.000557919797081233,-0.0009488693825509789,-0.001355944972092804,-0.001771857287096935,-0.0022347645969432203,-0.002699663420461693,-0.0031444812779033685,-0.003597298199898792,-0.004059531932011968,-0.0045289373352221255,-0.00496248833339629,-0.005403289064200292,-0.005918359760841455,-0.006453473964684188,-0.0069859010688933555,-0.007510609873085311,-0.008039637842617874,-0.008616027355715917,-0.009209856863049032,-0.009783340271668479,-0.010294320232271802,-0.010811785885011948,-0.011374372814089378,-0.01198211875538492,-0.01260909891759777,-0.013255598292036134,-0.013902250205397306,-0.014556877304306506,-0.015233043379504282,-0.015942001938819454,-0.016692313951551463,-0.01752511169149967,-0.01838957600502184,-0.019309611001573513,-0.02025227541910438,-0.021231381069356875,-0.022192268942857873,-0.023181478789764176,-0.024207075591498397,-0.02529498184801371,-0.026464521790015915,-0.02767492586488703,-0.02890738328356356,-0.030200595891009457,-0.03144121497620727,-0.03271965383423466,-0.03399764580432452,-0.035309719598069784,-0.03664635298892949,-0.038025632377017615,-0.03943111782506998,-0.04081932023287406,-0.04224277720937278,-0.04370349735327047,-0.04523120237099906,-0.046856938045392316,-0.048529538453671674,-0.050230165166855,-0.05198603615802797,-0.053769808777185034,-0.05554022208731415,-0.057316350563097807,-0.05914690761162041,-0.06094347143983408,-0.06267888943432327,-0.06443432701381097,-0.0662490815583961,-0.06812895990042057,-0.07000059418604787,-0.07190146956440568,-0.07381509853836925,-0.07576090859973332,-0.07777858414877216,-0.07978637396638456,-0.08174934687955847,-0.08368595710841632,-0.0856554932152193,-0.08770014731206162,-0.08976167880957579,-0.0918620320660664,-0.09399159573610472,-0.09615973281758623,-0.09837133925161919,-0.10055896450882046,-0.10273093634826523,-0.10492231732491225,-0.10716046086278334,-0.10938093792426733,-0.11156561683905686,-0.1137649402077624,-0.11599951571673667,-0.11825161184964843,-0.12053224056566976,-0.12278350161639921,-0.12500925715828068,-0.1272702563652031,-0.12956546029430085,-0.13182628560112392,-0.13402860989792867,-0.1362252496079331,-0.13846592837823066,-0.14067383298440547,-0.14284997386563786,-0.1450181538264582,-0.14724076883027964,-0.14949817437077742,-0.15179238805897402,-0.15403194412596546,-0.15625395097682776,-0.15849517273201696,-0.16070122542985618,-0.16291937405609316,-0.16510047259270294,-0.16728194600669308,-0.16947827719870504,-0.17165714573439364,-0.1738195163673951,-0.1759866578838354,-0.1780722273035741,-0.18014949600306052,-0.18225275508976732,-0.18435890419844345,-0.18643713134364914,-0.18852779191348765,-0.19062063097395066,-0.1927293885060972,-0.19481523645857243,-0.19684810403689798,-0.19885109741210685,-0.20077882700566704,-0.20261907341769073,-0.20440314073528063,-0.20614124409440462,-0.20779733193137448,-0.20939639626896897,-0.21091809035981302,-0.21238526918484338,-0.21389143647137,-0.215391639427411,-0.2168816208186565,-0.21843433696460055,-0.2200218519450423,-0.22155202028669044,-0.2229633455471457,-0.22435989138691845,-0.22572812693950972,-0.22701490711449457,-0.22828230920487533,-0.22954053031730237,-0.2307954886432474,-0.23206482574382256,-0.2333529048737946,-0.23459119953095553,-0.23581882602404272,-0.23707891336804143,-0.23827141200456914,-0.23940711570146886,-0.24044294090540527,-0.24145645333485088,-0.242500447663236,-0.24352475279937302,-0.24447821137762982,-0.24544389494187727,-0.24638394010296846,-0.24727211080235925,-0.2480785921720868,-0.2488700054359607,-0.24966356178076596,-0.2504710049203522,-0.25120984827506776,-0.25185122843119007,-0.2524676572562033,-0.2530781691400709,-0.25360908697128076,-0.2540477508714496,-0.2544081870027128,-0.2547373777820075,-0.25508870736609274,-0.25537909209728493,-0.25563637977245707,-0.25584785890878464,-0.25599642736127565,-0.25600798958469195,-0.25595046113676595,-0.25589916810591035,-0.25584945180480306,-0.25569722053265914,-0.25553088025875714,-0.2553994488491357,-0.25521907420023476,-0.25495914773927647,-0.2547398304637784,-0.25454224080077387,-0.25435764794148014,-0.2541745246527253,-0.25393507055576753,-0.25368705576929695,-0.253357511361689,-0.2529359999639553,-0.2524405239153344,-0.2518855500948222,-0.25128898247157644,-0.25060222490076717,-0.24979562442658695,-0.24887278113709835,-0.247856742459136,-0.24681018885722797,-0.24568646240754574,-0.24450237305464262,-0.24324700953131329,-0.24194350659097089,-0.24061935207222462,-0.23924500602288248,-0.23780421461537415,-0.2363300615090368,-0.2348865345399153,-0.2334764545432839,-0.23206055958420446,-0.2306462144753274,-0.22922285955929755,-0.2277953461194794,-0.22629843689037435,-0.22475606382449623,-0.2230980706092243,-0.22141619942951427,-0.2197885639217403,-0.21816733036072125,-0.21647856924150058,-0.21473638907900686,-0.2129626311938504,-0.2111540154492618,-0.2093905294870496,-0.20759096665779395,-0.20576510341256746,-0.20391909206629108,-0.20204897026972998,-0.20015979088884084,-0.19825102515852402,-0.1963276845109293,-0.19440816222542787,-0.19247233516129844,-0.19051509429744853,-0.18863635841636828,-0.18681531662228235,-0.185061733910695,-0.1833626314197542,-0.18165259265015452,-0.17999239887113663,-0.17821022866989397,-0.1763329046405466,-0.17438170040522008,-0.1723887731543222,-0.17039675451360417,-0.16835811133437664,-0.1662982360129744,-0.1641523307646411,-0.1619943345480985,-0.1597658699151398,-0.15759034040746286,-0.15538997866690793,-0.1532442514434114,-0.1510693455533842,-0.1488748387023673,-0.14668893292846594,-0.14456083418433,-0.14240299632935957,-0.14019847099402502,-0.1380199217608579,-0.13588752665947276,-0.13379272326316438,-0.13167818282769578,-0.1295383346117414,-0.12742824606576644,-0.12534142626488545,-0.12330425653015557,-0.12127435881606652,-0.1192440339153305,-0.11721077392967398,-0.11521803567392208,-0.11324569036474619,-0.11127469287141463,-0.10926592277968854,-0.10725440915645092,-0.1052903411549732,-0.10333018969709488,-0.10141562420095428,-0.09946036276853454,-0.09754301253778147,-0.0956500224513603,-0.09374008077332047,-0.09178974166154145,-0.08988730842671128,-0.08799073102536153,-0.08606693852269852,-0.08419899507022353,-0.08235508097570234,-0.08055955782771142,-0.07876784873051593,-0.07704069829330697,-0.0753161131280215,-0.0735696846042976,-0.07185371109093813,-0.07019089436361536,-0.06856649927203341,-0.06697882882343437,-0.06546000302898361,-0.06401317588990672,-0.06259218390560946,-0.061166166188948594,-0.059728649535884026,-0.05830250634093556,-0.056886722400308704,-0.055458583965968276,-0.0539773786824011,-0.05244363828335628,-0.05091497375980602,-0.049391384524822286,-0.04792145150071339,-0.04648216725180047,-0.04504296665590298,-0.043654395711229595,-0.042329479650705326,-0.04106616504587505,-0.039801358729615925,-0.03854133943252102,-0.03729002523806511,-0.036065657996289996,-0.034875532220671905,-0.03379857806972648,-0.03271808858990048,-0.03160532441849274,-0.03051747917375493,-0.029470697777898375,-0.02838345275484257,-0.027303700744277737,-0.02618907278282652,-0.02513836426712958,-0.024106590437341674,-0.02311921460616104,-0.02216314415502192,-0.02118966296301247,-0.02020957516362099,-0.019298746447046728,-0.018349054936849647,-0.01741368412814474,-0.016535996760981362,-0.015683465511326114,-0.01481846091089784,-0.01400497960799381,-0.013198830867187162,-0.012384196490909723,-0.011599484352593392,-0.010815627527882099,-0.010016749293817686,-0.009197063781281987,-0.00841828936443659,-0.0076700288720924145,-0.006919936618309401,-0.006123198845190748,-0.0052860575670252785,-0.004339288608694153,-0.0032790768737059454,-0.002188446959957494,-0.0009854420327641304,0.00023148883783199382,0.0072827090545313665,0.008543363923673443,0.009790962082062878,0.0110095804374691,0.012178915206097964,0.01338038337688819,0.01461092170870508,0.01575514659899229,0.016897776971103496,0.01804843812145682,0.01926054217365447,0.02048992919422903,0.021736328000865387,0.0229951218039246,0.024385877569344266,0.02588763356811936,0.027525219548488352,0.029244382860016984,0.031076055302033334,0.03305628889223571,0.03525700530387965,0.03755314580153172,0.039909303966214345,0.04232257244208177,0.0449292858731731,0.04759331102570879,0.050322050282393234,0.05314735813391713,0.05602577289353147,0.05890336728066894,0.06188020243259687,0.06491822911326638,0.0680533088846835,0.07125303007708254,0.07452915644723555,0.07784147269049328,0.08128691420210499,0.08478692236400102,0.08835812575980978,0.09192126926073979,0.0955421956212319,0.09912321711310505,0.10249732193189576,0.10586164520475894,0.10914785556036417,0.11242063762768305,0.11560974467142625,0.11876715322380971,0.12197001481635694,0.1252022414705084,0.12840875468922933,0.13165989116673188,0.13494114697086076,0.13824260345868253,0.14159633125129847,0.14503154409647592,0.14852710264731583,0.15205314508099796,0.15548413054453394,0.15887057002278435,0.16217298177878622,0.16532488772856294,0.16838600316068145,0.17132039055153025,0.1740268050230814,0.1766488106436075,0.1792265281532178,0.18176504673142257,0.18428303981428773,0.1868422940929683,0.18946166224248004,0.19206190089761765,0.1946817881300903,0.19739404519927672,0.20024263939268602,0.20313528968227476,0.20608172358700647,0.20900807585352607,0.21195429942808897,0.21475051369339837,0.21733384395776412,0.21981790696763806,0.22225004217476582,0.2244428167962582,0.22650328940649392,0.22844526565203369,0.2303319117784746,0.23216714462882604,0.2340581156803918,0.23591812262129122,0.2378677970809214,0.23977797671742007,0.24173589388183847,0.24390612494097563,0.2460494586312086,0.24816362074751477,0.2503405255909456,0.2524846000692183,0.25453152445471927,0.25654518497871875,0.25834899404510697,0.2600501037423147,0.26155144350286175,0.2629008955499767,0.2640657215017514,0.26505014996260423,0.2659313846520558,0.2667376893467585,0.26743819125751095,0.2680417069890147,0.2686336067557166,0.2693191461048136,0.2699956390986357,0.2706778643483821,0.2714911701347684,0.27246334969707703,0.2734183069793326,0.274322771429021,0.27531683139540253,0.2762151674817234,0.27709867192806553,0.2776667881611935,0.278099693258088,0.27843345535080083,0.27873554338559225,0.2789185671694286,0.2789736691642499,0.27892992552099566,0.2788123396763366,0.27857757937937394,0.2781604927752146,0.2775965760924595,0.27705061744496035,0.2765217196658,0.2759004091364061,0.2753168524696723,0.27475801247518855,0.27418988442003117,0.27358439639623594,0.2728644153904394,0.2719541153961408,0.271022103502726,0.2699634221554618,0.2687970439825281,0.26756380975912897,0.26629793714454364,0.2651183513695436,0.2639858912632494,0.2627290356449277,0.2612641938962614,0.25967480998543174,0.25807595026989133,0.256486417820653,0.254978060569764,0.25344102114447087,0.2518058197725346,0.25017149979369585,0.24861071991400996,0.24704797812425802,0.2454910271927738,0.24391086823272792,0.2422442078056628,0.24054234181406386,0.23881493867743667,0.2369531458312284,0.23507690207366363,0.23321822333957057,0.2312910471785271,0.22935113575792107,0.22732262493729694,0.22528627310618754,0.2231819294977181,0.2209652525323186,0.21878764502836048,0.2166638677756607,0.21447398812624194,0.21224494072657782,0.20997802366756071,0.20768134748812975,0.2054706745454977,0.20327314374317476,0.20112565870155363,0.1990201518835849,0.19694537725480185,0.19491196042348888,0.1927742645020111,0.1906357753418776,0.18846315478788125,0.1862951179017739,0.18398347743322788,0.18163406943039992,0.1794224144972435,0.1772874559287959,0.17507254593650737,0.17283796188532696,0.17063304436218887,0.16841835003077377,0.1661002821407245,0.1637139220499146,0.16135483653149668,0.15900303439504407,0.15673846895392368,0.15449382357761535,0.15231033839033017,0.1502244317538424,0.14823034387223047,0.14626168454243968,0.1444461248512174,0.14272308668641948,0.14094048435949671,0.1391999898524204,0.13748769059072857,0.13569783049848316,0.13385660958287451,0.13194533906306546,0.1299460318261997,0.1278804799664661,0.1257888123803472,0.12375363440283839,0.12170462774943423,0.11965133311946931,0.1176716616271166,0.1156965234130737,0.11367645499049568,0.11173078531893034,0.1098650160184399,0.10805891810605615,0.10627509894616503,0.10442666445659504,0.10266841860883173,0.10097537839450182,0.09935962608385611,0.09783540123681651,0.09631986738911356,0.09477561638023496,0.09326383156760137,0.09170830736939718,0.09020615667454879,0.08871185497958528,0.0871685621984471,0.08564238598994332,0.0841669327168072,0.0826700458047584,0.08116170564044116,0.07965701246151936,0.07821826312187033,0.07685961339074793,0.07553915804370738,0.07420775408737558,0.07288067368561639,0.07162855901796848,0.07042719514118313,0.06919816994621884,0.06804895088037971,0.06688387030334926,0.06565561374947955,0.06442754639493761,0.06326176862806553,0.0621294881526265,0.06099236184935027,0.059795649423489396,0.058587842702635645,0.05734210992154492,0.05606375854510445,0.05482925546370255,0.053624944118924746,0.052459215424284396,0.05134300563862481,0.050271107223235856,0.04920030288068713,0.048152316401315655,0.04714816302460051,0.04611277354077978,0.04510665111741081,0.044127291381389544,0.043151892845765545,0.042241892838306094,0.041368604284897574,0.040475131480563586,0.039594237101833525,0.03871146306654627,0.03790110838162482,0.037127781383834955,0.03637921107815124,0.035590678517483314,0.03480475220832155,0.034019215459257894,0.03321885341065222,0.032428534893646964,0.03169186769375753,0.031006039942484978,0.030373541756379793,0.029787148574535516,0.029194314193595767,0.028583296976598425,0.02796131000788805,0.027336258942836866,0.026746205182135368,0.026152942544712055,0.025566879646805957,0.024951821275207894,0.024353725803704058,0.02380941778715694,0.023325367037938857,0.022824476227067764,0.022325973653569017,0.02180636195118898,0.021226787243411413,0.020652362408356814,0.0200800803937685,0.019558332287526212,0.019079851261772963,0.018609782350995363,0.018139855947652307,0.01766204358263324,0.01720661742626391,0.016743868499090882,0.016304378081989552,0.01587958671834795,0.015527024378278657,0.015197338555718254,0.01484510631020318,0.014469896927176063,0.014118918905515592,0.013793219040049614,0.01345871456714661,0.013089983052930314,0.012722526842170012,0.012342917433799137,0.011942891227578816,0.011581212906372705,0.011245917068081942,0.010877509136186759,0.010473654828063264,0.010066219464202985,0.009622247866513368,0.009188412757833452,0.008814413863921395,0.008434972843943084,0.008042835071221442,0.007615905691975108,0.00726059078303654,0.006906170347442939,0.006546384107452661,0.006262299557191542,0.005976449412395478,0.005660525460378499,0.0052781015429243525,0.004894432631577576,0.004527946696896638,0.004224012972759482,0.003932353586377909,0.00364369881081378,0.0033697373542699543,0.0030615083615664566,0.0027160351123380077,0.002361236702433053,0.002043980675963711,0.0017532926184951811,0.0014672896031936028,0.0011785302746184926,0.0008956071104900218,0.0005744744864531028,0.00023276822910941122
    ]

    ACT32_LEFT_HAND_VEL = [
        0.0001665219997427954,-0.00021035796573859534,-0.0005753090961949401,-0.0009115776339112764,-0.0012202120930537637,-0.0015325711844277137,-0.001825462698772768,-0.0021261181221184744,-0.0024439356076379364,-0.0027825881429826174,-0.003173686083945089,-0.0036073922090986304,-0.004014428996245536,-0.004406309273040223,-0.00475711372098864,-0.005102525397390314,-0.005468897996391161,-0.00582530205878543,-0.006217607302562954,-0.006653303189072794,-0.007127814774423553,-0.007610304433775677,-0.008143647674862479,-0.008684799357705112,-0.009173449143709983,-0.009652958733122824,-0.010162834798064022,-0.010688017474204925,-0.011238075114145788,-0.011843747773691652,-0.012484673758421688,-0.013155338586559854,-0.013812723575370695,-0.01446374079964717,-0.015154458185552352,-0.01587153647414384,-0.0165752170056174,-0.017288878318368936,-0.018041233032814623,-0.018826941896012517,-0.01961345892613554,-0.02041724701651537,-0.021179311563271148,-0.021928027986189896,-0.022701432251973953,-0.023548489684468044,-0.02442658773506644,-0.025290099520194703,-0.02616136688640335,-0.027102273947512325,-0.028069411988417007,-0.029087942570766117,-0.03013856858262018,-0.03119558042985726,-0.032264112928826784,-0.03337250217212036,-0.034524106956800645,-0.03570779893349695,-0.03686545408984784,-0.03806824323912014,-0.03931990940689576,-0.04063342671824902,-0.04200172236859564,-0.043405732212777985,-0.04482785803730843,-0.04627858929010884,-0.0477241652043068,-0.04913190023775329,-0.05061514202374436,-0.05212653082829712,-0.05361097824336252,-0.0551449587244405,-0.05670109068735089,-0.0583021536965746,-0.059885040715986715,-0.06148776633021058,-0.06312386027995863,-0.06478426199782829,-0.06644285548082225,-0.06811228082832389,-0.06982820502783739,-0.07155006484167033,-0.07329929344155037,-0.07508108751313253,-0.07687812132824111,-0.07870512795558376,-0.08058547414845575,-0.08255136936030996,-0.08453486998101382,-0.08646804233455167,-0.08840706950010153,-0.09037415723493632,-0.0923230128575933,-0.09422638715133062,-0.09612960234038499,-0.09806632040770079,-0.10007037639791917,-0.10208798564685455,-0.10409830023675004,-0.10615470449971848,-0.10822767353952217,-0.1103083289341379,-0.11236981554852864,-0.11443101832251772,-0.1165351516353682,-0.11866732838747847,-0.12084962568724354,-0.12304286473124566,-0.1252605207130789,-0.12748588562150523,-0.12972366680921313,-0.1319504137464441,-0.13414989597541943,-0.13634072000264397,-0.1385256245192323,-0.14070758719021528,-0.14281356767989503,-0.1449177809385623,-0.14698862079837102,-0.14905291594184392,-0.15111724547286545,-0.15316483941013276,-0.1552488586394737,-0.15732299473600747,-0.1594334825014382,-0.16157896150770204,-0.16371545687369593,-0.1658303811026112,-0.16796685616275495,-0.1701282070587824,-0.17230835611868034,-0.17452831646017733,-0.1767832041719185,-0.17905750636392856,-0.18127142085708772,-0.183421593221329,-0.18552500967153165,-0.18758886383676496,-0.1896066916667315,-0.19156524542907982,-0.19346665266392038,-0.19535942958173644,-0.1972736828466617,-0.19920398948007226,-0.20116748813299545,-0.20313358086738945,-0.20497840079345372,-0.20677705000426252,-0.20860561310670644,-0.21047176068253073,-0.21231350081087863,-0.2141425581602735,-0.21592706713149248,-0.21769325348308408,-0.21943141107127642,-0.2211821186619183,-0.2229345954511404,-0.224644312278141,-0.22625933190828573,-0.22791736047968916,-0.22962052340647,-0.23127567105053773,-0.23288815028243293,-0.2344635975978412,-0.2360134978984343,-0.23737193507607088,-0.23870165470764607,-0.2400116591286907,-0.24131550466164023,-0.24259030616857832,-0.24382846509217496,-0.24502865678110322,-0.2462108599410219,-0.24738073909116368,-0.24845151993371703,-0.2494671642493822,-0.2504382762888896,-0.2513774982189377,-0.25225163000511475,-0.2530681577272671,-0.25378237424287287,-0.25449323412374697,-0.2551370372789324,-0.2557460300177621,-0.2562115080076158,-0.256638121686959,-0.25699527077924106,-0.25731925359230645,-0.2575919943062011,-0.25781213627567073,-0.2580066660004723,-0.2582107621906558,-0.25837244017551625,-0.2584802729843286,-0.258573002362882,-0.25863863206738097,-0.2586905928954214,-0.25863613911902344,-0.2584893928491311,-0.258223784185838,-0.25797232258650066,-0.25766167712541777,-0.2573210124835261,-0.2568951645921971,-0.2565075579567578,-0.256152434396094,-0.25582035340389425,-0.25540966633407836,-0.2548345742666102,-0.25416411779524695,-0.2534103402085514,-0.25261445128231524,-0.2518045766882075,-0.25096431069129005,-0.25007721380056475,-0.2491402409802822,-0.2481709448660312,-0.24711865533286478,-0.24602629517332109,-0.24488122886251473,-0.2436803597123853,-0.24239919491551365,-0.24109371141095234,-0.2397804182523803,-0.23849129242626757,-0.23714888516016544,-0.235799708619472,-0.23441669438087454,-0.23297417184156854,-0.23149953318113706,-0.2300016336497822,-0.22844665741240755,-0.22684702207074625,-0.22526675150748474,-0.223680486844016,-0.22202405085133714,-0.22034901878131352,-0.21870918981642498,-0.21699798390747546,-0.21521808067394377,-0.21343340083055293,-0.21161606798202928,-0.20972577058189898,-0.2077145840996537,-0.205685799062035,-0.20360976360411318,-0.20151705770772535,-0.19945867623659816,-0.19734790657850618,-0.19524900443458656,-0.19319450972833682,-0.1911629529775469,-0.18913525123514477,-0.18712066912806327,-0.1850744477288617,-0.18303478843539575,-0.18102603808630105,-0.17905262472009587,-0.17713186807295403,-0.1752183880287664,-0.1732875000168664,-0.17136598285276117,-0.16942860512149371,-0.16742452533612304,-0.1654136997432313,-0.1634313030741223,-0.1615237530619648,-0.15969736659530312,-0.15787206363216957,-0.15601439821743393,-0.15411744218819665,-0.15212464383593063,-0.15012376762990617,-0.14813932568624538,-0.1461848156840261,-0.14427462481445782,-0.14234352103774997,-0.1404322854407636,-0.13846503722501466,-0.13645066812988005,-0.13440109259784774,-0.1323761637031057,-0.13036047878204124,-0.1284062574483967,-0.12657856414356808,-0.12480733673856248,-0.12308729711338023,-0.12136272914849927,-0.11962401271123875,-0.11789492909307318,-0.11612141665315824,-0.11434016885653976,-0.11260152882240482,-0.11087874756038753,-0.10913880291449339,-0.10727675249487077,-0.1054625195536185,-0.10370139712211375,-0.10200615088874328,-0.10035368519995513,-0.09868444448726595,-0.09703886380309969,-0.09535791309154223,-0.09369939223432742,-0.09200637020477376,-0.09029287666467123,-0.08860107894298275,-0.08699864766838046,-0.08546086497857425,-0.08405366138980626,-0.0827381007134234,-0.08134990114010568,-0.0798815721328587,-0.07839942651300104,-0.07682046729315238,-0.0752425099897858,-0.07367481049777079,-0.0720696405162946,-0.07043849154516878,-0.06886092416430213,-0.06730173784987097,-0.06579641377244241,-0.06426246248316701,-0.06268746230482948,-0.06114667138883817,-0.059617568463133576,-0.05810657063222938,-0.05661086588862596,-0.0550804712725076,-0.05361625307221699,-0.052224704533560345,-0.05090641758104924,-0.04971704061343712,-0.04854209221894649,-0.047296943880393356,-0.04604634543864127,-0.044861784159001106,-0.04376502623780118,-0.04266732360696248,-0.041549973451226824,-0.0404618931272071,-0.039390930698070654,-0.038292091015282116,-0.03717508740892616,-0.036017734786332724,-0.03482691105706728,-0.03355163433120547,-0.0322053741508723,-0.030850483165957163,-0.029498387034067347,-0.028164979793895154,-0.026812264569917036,-0.02551449946720102,-0.024169592627457732,-0.02272785820553719,-0.021336263220687126,-0.0199892797408782,-0.018701726212051368,-0.017408473606753006,-0.016145500288543318,-0.014870618519551444,-0.013547475973658761,-0.012166329291186046,-0.010743333561230204,-0.009261070974679349,-0.007733425025400594,-0.006126878043660608,-0.004526485610307958,-0.0029212515381453683,-0.0012625019377847835,0.00374563241096173,0.0102680461246896,0.011755846791519716,0.01322622803229325,0.014639111635522771,0.015972273191759496,0.01714276126360514,0.018383128210633034,0.01966911640295627,0.020961548506700105,0.02239939021526367,0.023901246721701807,0.025436788040646297,0.026930102577543472,0.028432702805874576,0.030014881701939212,0.03158721085665599,0.033203063243755986,0.03482549144697316,0.03649233556273507,0.038178569393435974,0.039853266367474834,0.04160295911101159,0.04343468867702172,0.04530811927029207,0.047221230797047375,0.04924688231917055,0.05148509259424441,0.05376647567318735,0.056091691132560954,0.05846817266897425,0.060879313868160166,0.06344846611592495,0.06610454042622292,0.06880253651492377,0.07167005324835986,0.07460921120486051,0.07764558232991445,0.08075306890444946,0.0839279445563985,0.08708597641244718,0.09029272591416578,0.09354498500499654,0.09684599026806652,0.10021775478935911,0.10368123010303369,0.10713891832862643,0.11063114221641153,0.1141128371753517,0.11765116215362607,0.12123697909645124,0.12490609899383616,0.12863964645318518,0.1323570532373234,0.13605792159430777,0.13963580143677248,0.14317083291712468,0.14665557845964997,0.15016372963768337,0.15383920657027517,0.15747274584272433,0.16110411111056935,0.16468188650221366,0.16828425554770723,0.17200985587302062,0.1757090144457415,0.1793137783726653,0.18286148671969357,0.18635127189909936,0.18981127593023164,0.1931966652999261,0.19635159460026594,0.19938881204436984,0.20229758802786194,0.2050905133837548,0.20787503031098128,0.21056950717036024,0.21320984384828953,0.2158020823954835,0.21830587358675713,0.22071338126552545,0.22313259049336628,0.2254916881766147,0.22779706263606148,0.2301790403954652,0.2325220558945376,0.23470507877900457,0.2367344477257077,0.23878271821315508,0.24074196332436168,0.24273889391097558,0.24463838270609636,0.2465057621445705,0.2481894985234757,0.249770129160278,0.25125596910010967,0.25273546536611496,0.2541746477809235,0.25565742616295045,0.2571833800828555,0.2586474403593208,0.2600074583905858,0.2612901746016933,0.26245579947611347,0.2634189779793751,0.2643040442167346,0.26510360378935655,0.2659449559435929,0.26693036840285855,0.26794318939590844,0.2689236339176171,0.2698830175285033,0.2708772114904348,0.27174288581283806,0.2725664742704339,0.2733090584547488,0.273918478961222,0.27438942061223753,0.2747067706544986,0.2748475332381723,0.27479556584386217,0.27468549808586257,0.27440764006979373,0.2740939750373977,0.27377451141528547,0.27345615899209885,0.27311629818658223,0.2727366404794442,0.27237611593157535,0.27193055338271915,0.2714174216435838,0.27085630770761626,0.27025756305462756,0.26963862489064777,0.26901236318090765,0.26842331559410987,0.2677759422973381,0.26710094691092356,0.26642723652941713,0.2657523969618027,0.2649080758763502,0.2640015568452291,0.26296698758607495,0.261793798551905,0.26051802455998035,0.2589984998294339,0.2574225119423893,0.25590221359866294,0.2544342563462438,0.25302988715706215,0.25161879885289756,0.2502335982491554,0.24875348707680123,0.247135245063388,0.24556324750070554,0.24408811846762699,0.2425975115227669,0.24112629552661183,0.23969819656175792,0.23814992245943234,0.23662806316527732,0.2350897465243418,0.23339082020019433,0.23160027415519335,0.22981050003038228,0.2280243827630752,0.22628494473755292,0.22443738491203757,0.22253506067082654,0.22061668258071435,0.2187803953972659,0.2169979327160967,0.2151463479348246,0.21332999646336812,0.21169104183361562,0.21006896502038042,0.2084492034234728,0.20688608934498318,0.2052295452669947,0.20345728542656208,0.20161981239327795,0.19977035152506548,0.1979820938071435,0.19610182521708786,0.1941065151798411,0.19213967229705317,0.19013080864876672,0.1880507254755791,0.1859351770014055,0.18386071980800753,0.18181271766493876,0.17973360668206181,0.17766961938350131,0.17562927798391073,0.1736058855997429,0.1715444708266847,0.16944711029992868,0.16730204669034301,0.1651008484174461,0.16295770191536152,0.16079714718708912,0.158633685387339,0.1563643088911837,0.15406930242033595,0.15181806657024785,0.14951587264031832,0.14735996084427835,0.14525919422954195,0.1431408754801372,0.1409827086471853,0.1388504177109725,0.13677335038153923,0.13474780953013468,0.13271160868986706,0.13069741885977887,0.12863641185379882,0.12658985388316,0.12460158092344786,0.12269530656336607,0.12083454700194222,0.11900888135348847,0.1171569107950547,0.11536720071747637,0.11366638641666393,0.1120125410035977,0.11038596579814569,0.10876296492396657,0.10711179783873077,0.10540236867190987,0.10366233217825825,0.10194338201061841,0.10033700205261441,0.09865648363022478,0.09697809887233322,0.09537553713510234,0.09384776715755501,0.09237788579703003,0.09093391794224708,0.08953668108092667,0.08819661125467286,0.08689056707425401,0.08564050108776987,0.08442299135149921,0.08321446841334365,0.0819599081204787,0.08069891244833712,0.07952532850357986,0.07836471560693334,0.0771971364576834,0.07600294458416694,0.07478797934450596,0.07345669074334929,0.07207088479200702,0.0706239472102988,0.06918180976701233,0.06776185976861535,0.06637354986577075,0.06508089634265783,0.0638321167681148,0.06265860402303618,0.061621876383689905,0.060607350210073185,0.05962052186051486,0.05860649768772371,0.057586532996815064,0.05660459064304591,0.0556383009059363,0.05461862249040001,0.05362646581302798,0.0527036763256566,0.051814659975670335,0.05092594267047173,0.050037063913345114,0.049163261000392235,0.048234925386694445,0.04722995296245704,0.046164157791852296,0.04509537600982369,0.0439880343088588,0.0429544641923324,0.04206824452033576,0.04125503975189352,0.04047278760632874,0.039677275748698414,0.03888371731436047,0.038111997152053476,0.037362346924267015,0.03662497341718844,0.03587098930785144,0.0351162148243159,0.03435382744130046,0.033677746937940935,0.03309484444109725,0.03252941562800966,0.03196417023180131,0.03134723100221195,0.03075033021859966,0.03019148043487599,0.029637075929442775,0.029096640641249092,0.028460917953797096,0.02778823258322145,0.027157474790355865,0.02657352413285072,0.02615027200650806,0.025774564826288312,0.02537749790459719,0.024970873225097515,0.024547245768794784,0.024118516031840603,0.02364092465354663,0.023104745085129705,0.02251407164133444,0.02190009363193723,0.0213051079696058,0.020717857918901194,0.020115991148965475,0.01955166471350082,0.01900900319499304,0.018473249424356436,0.017927686523444043,0.017403618894074116,0.016873299757142923,0.016352151695382046,0.01581619576612883,0.015221360525742397,0.014618429643084586,0.014036614660114285,0.013477301597512403,0.012978489441950526,0.012512932246736658,0.012086664497205047,0.011678175523285898,0.011313491160257884,0.010946970262514719,0.01053835152471165,0.010115619008648694,0.009720445298811707,0.00935508028075885,0.008964074509396071,0.00852931156891483,0.008081468951449084,0.007620128637651973,0.007198199429176953,0.006792683076453511,0.006410503700579145,0.005972559563283717,0.005542611302655408,0.005109812686832943,0.004672909017821333,0.0042231764798489994,0.003755623888315659,0.0032765126998670884,0.0028109184033257204,0.0023779867137403267,0.0020122547969581143,0.0016709425104254232,0.0013147574657343625,0.0009367308528254583,0.0005599391421851748,0.0001665219997427954
    ]

    # --------------------------- ACTIVITY 1 ---------------------------

    activity1_indexes_per_healthy_A_PRE = [
        # Participant 1
        [
            [(0,42), (43,99), (109,151), (152,206), (207,258), (259,313), (314,367), (368,422), (423,478), (479,537)], 
            [(0,43), (44,92), (93,142), (143,190), (191,241), (242,290), (291,342), (343,397), (398,445), (446,499)]
        ],
        # Participant 2
        [
            [(0,52), (53,107), (108,162), (163,218), (219,272), (273,326), (327,380), (381,435), (436,488), (489,537)],
            [(0,46), (47,96), (97,146), (147,197), (198,249), (250,302), (303,355), (356,407), (408,458), (459,499)]
        ],
        # Participant 3
        [
            [(0,53), (54,119), (120,182), (183,242), (243,299), (300,358), (359,417), (418,476), (477,537), (538,617)], 
            [(0,62), (63,120), (121,176), (177,235), (236,294), (295,352), (353,412), (413,476), (477,533), (534,594)]
        ],
        # Participant 4
        [
            [(0,37), (38,72), (73,107), (108,142), (143,178), (179,216), (217,254), (255,290), (291,330), (331,369)], 
            [(0,38), (39,68), (69,107), (108,142), (143,173), (174,212), (213,247), (248,284), (285,318), (319,358)]
        ],
        # Participant 5
        [
            [(0,64), (65,122), (123,181), (182,238), (239,291), (292,343), (344,397), (398,455), (456,509), (510,580)], 
            [(0,43), (44,93), (94,143), (144,207), (208,250), (251,310), (311,355), (356,409), (410,462), (463,518)]
        ],
        # Participant 6
        [
            [(0,44), (45,89), (90,136), (137,183), (184,230), (231,278), (279,325), (326,373), (374,419), (420,470)], 
            [(0,45), (46,92), (93,139), (140,189), (190,237), (238,288), (289,336), (337,386), (387,436), (437,490)]
        ],
        # Participant 7
        [
            [(0,44), (45,104), (105,157), (158,211), (212,260), (261,319), (320,376), (377,430), (431,486), (487,530)], 
            [(0,53), (54,111), (112,159), (160,219), (220,286), (287,340), (341,400), (401,450), (451,506), (507,561)]
        ],
        # Participant 8
        [
            [(0,39), (40,83), (84,128), (129,175), (176,223), (224,264), (265,308), (309,353), (354,399), (400,451)], 
            [(0,40), (41,80), (81,125), (126,168), (169,214), (215,257), (258,302), (303,347), (348,392), (393,462)]
        ]
    ]

    activity1_indexes_per_healthy_A_POS = [
        # Participant 1
        [
            [(0,52), (53,108), (109,164), (165,218), (219,272), (273,326), (327,379), (380,433), (434,486), (487,547)],
            [(0,43), (44,97), (98,148), (149,206), (207,262), (263,318), (319,375), (376,442), (443,507), (508,563)]
        ],
        # Participant 2
        [
            [(0,62), (63,121), (122,182), (183,240), (241,298), (299,356), (357,415), (416,475), (476,532), (533,596)],
            [(0,47), (48,100), (101,155), (156,211), (212,266), (267,319), (320,374), (375,428), (429,482), (483,530)]
        ],
        # Participant 3
        [
            [(0,70), (71,141), (142,212), (213,283), (284,350), (351,420), (421,494), (495,554), (555,633), (634,699)], 
            [(0,46), (47,106), (107,167), (168,227), (228,288), (289,354), (355,411), (412,471), (472,536), (537,595)]
        ],
        # Participant 4
        [
            [(0,41), (42,79), (80,119), (120,162), (163,204), (205,246), (247,286), (287,327), (328,370), (371,413)], 
            [(0,37), (38,73), (74,114), (115,153), (154,194), (195,236), (237,278), (279,319), (320,359), (360,405)]
        ],
        # Participant 5
        [
            [(0,72), (73,139), (140,215), (216,289), (290,360), (361,432), (433,503), (504,575), (576,640), (641,713)], 
            [(0,69), (70,131), (132,201), (202,270), (271,328), (329,395), (396,461), (462,520), (521,573), (574,638)]
        ],
        # Participant 6
        [
            [(0,51), (52,108), (109,164), (165,215), (216,266), (267,317), (318,374), (375,428), (429,479), (480,530)], 
            [(0,48), (49,100), (101,152), (153,202), (203,255), (256,307), (308,360), (361,412), (413,469), (470,524)]
        ],
        # Participant 7
        [
            [(0,48), (49,116), (117,164), (165,234), (235,283), (284,356), (357,417), (418,478), (479,526), (527,589)], 
            [(0,46), (47,105), (106,165), (166,227), (228,288), (289,351), (352,413), (414,477), (478,536), (537,599)]
        ],
        # Participant 8
        [
            [(0,44), (45,87), (88,131), (132,175), (176,223), (224,272), (273,320), (321,368), (369,416), (417,477)], 
            [(0,43), (44,92), (93,139), (140,187), (188,234), (235,279), (280,332), (333,377), (378,428), (429,487)]
        ]
    ]

    activity1_indexes_per_healthy_B_PRE = [
        # Participant 1
        [
            [(0,70), (71,131), (132,189), (190,252), (253,311), (312,374), (375,434), (435,493), (494,555), (556,618)], 
            [(0,49), (50,98), (99,144), (145,200), (201,254), (255,301), (302,356), (357,410), (411,474), (475,526)]
        ],
        # Participant 2
        [
            [(0,64), (65,130), (131,200), (201,265), (266,327), (328,394), (395,462), (463,526), (527,588), (589,661)], 
            [(0,55), (56,119), (120,177), (178,240), (241,304), (305,374), (375,433), (434,498), (499,559), (560,631)]
        ],
        # Participant 3
        [
            [(0,75), (76,139), (140,197), (198,271), (272,335), (336,402), (403,461), (462,528), (529,596), (597,685)], 
            [(0,52), (53,120), (121,185), (186,248), (249,307), (308,370), (371,433), (434,494), (495,557), (558,617)]
        ],
        # Participant 4
        [
            [(0,32), (33,69), (70,110), (111,148), (149,189), (190,226), (227,261), (262,300), (301,337), (338,386)], 
            [(0,36), (37,68), (69,104), (105,139), (140,174), (175,210), (211,243), (244,282), (283,318), (319,354), (355,405)]
        ],
        # Participant 5
        [
            [(0,54), (55,119), (120,178), (179,242), (243,304), (305,369), (370,432), (433,498), (499,563), (564,642)], 
            [(0,65), (66,130), (131,195), (196,259), (260,315), (316,384), (385,446), (447,511), (512,573), (574,640)]
        ],
        # Participant 6
        [
            [(0,61), (62,115), (116,166), (167,219), (220,271), (272,324), (325,377), (378,432), (433,479), (480,540)], 
            [(0,53), (54,107), (108,160), (161,216), (217,269), (270,328), (329,384), (385,439), (440,490), (491,551)]
        ],
        # Participant 7
        [
            [(0,49), (50,90), (91,139), (140,192), (193,244), (245,298), (299,365), (366,420), (421,468), (469,520)], 
            [(0,50), (51,95), (96,149), (150,200), (201,250), (251,298), (299,352), (353,403), (404,456), (457,512)]
        ],
        # Participant 8
        [
            [(0,48), (49,92), (93,138), (139,185), (186,234), (235,283), (284,333), (334,383), (384,434), (435,486)], 
            [(0,35), (36,76), (77,118), (119,159), (160,203), (204,246), (247,290), (291,332), (333,375), (376,424)]
        ]
    ]

    activity1_indexes_per_healthy_B_POS = [
        # Participant 1
        [
            [(0,43), (44,100), (101,150), (151,209), (210,270), (271,329), (330,391), (392,452), (453,514), (515,575)], 
            [(0,57), (58,115), (116,180), (181,237), (238,303), (304,367), (368,429), (430,492), (493,555), (556,620)]
        ],
        # Participant 2
        [
            [(0,58), (59,117), (118,180), (181,247), (248,306), (307,366), (367,425), (426,488), (489,548), (549,606)], 
            [(0,50), (51,104), (105,166), (167,225), (226,290), (291,356), (357,416), (417,480), (481,541), (542,604)]
        ],
        # Participant 3
        [
            [(0,67), (68,130), (131,193), (194,255), (256,313), (314,377), (378,439), (440,502), (503,566), (567,635)],
            [(0,58), (59,119), (120,179), (180,245), (246,306), (307,366), (367,427), (428,486), (487,547), (548,613)]
        ],
        # Participant 4
        [
            [(0,36), (37,70), (71,106), (107,146), (147,181), (182,218), (219,256), (257,293), (294,330), (331,375)], 
            [(0,36), (37,72), (73,108), (109,146), (147,181), (182,218), (219,254), (255,293), (294,331), (332,375)]
        ],
        # Participant 5
        [
            [(0,73), (74,144), (145,206), (207,275), (276,342), (343,414), (415,480), (481,547), (548,609), (610,672)], 
            [(0,57), (58,112), (113,171), (172,232), (233,294), (295,354), (355,417), (418,479), (480,535), (536,600)]
        ],
        # Participant 6
        [
            [(0,54), (55,109), (110,162), (163,215), (216,270), (271,334), (335,399), (400,459), (460,527), (528,602)], 
            [(0,59), (60,124), (125,193), (194,264), (265,330), (331,401), (402,471), (472,541), (542,606), (607,689)]
        ],
        # Participant 7
        [
            [(0,45), (46,96), (97,146), (147,199), (200,254), (255,307), (308,360), (361,417), (418,469), (470,519)], 
            [(0,43), (44,92), (93,143), (144,194), (195,248), (249,302), (303,356), (357,409), (410,464), (465,516)]
        ],
        # Participant 8
        [
            [(0,43), (44,84), (85,129), (130,174), (175,220), (221,263), (264,308), (309,356), (357,405), (406,456)], 
            [(0,35), (36,73), (74,116), (117,160), (161,203), (204,247), (248,291), (292,338), (339,383), (384,440)]
        ]
    ]

    activity1_indexes_per_healthy_C_PRE = [
        # Participant 1
        [
            [(0,45), (46,93), (94,141), (142,190), (191,242), (243,295), (296,345), (346,395), (396,454), (455,506)], 
            [(0,46), (47,95), (96,146), (147,201), (202,255), (256,308), (309,361), (362,415), (416,471), (472,523)]
        ],
        # Participant 2
        [
            [(0,72), (73,145), (146,214), (215,285), (286,356), (357,423), (424,488), (489,555), (556,622), (623,674)], 
            [(0,60), (61,116), (117,180), (181,233), (234,286), (287,342), (343,410), (411,459), (460,523), (524,571)]
        ],
        # Participant 3
        [
            [(0,70), (71,137), (138,203), (204,281), (282,343), (344,410), (411,476), (477,545), (546,610), (611,675)], 
            [(0,59), (60,138), (139,207), (208,279), (280,349), (350,415), (416,486), (487,553), (554,618), (619,685)]
        ],
        # Participant 4
        [
            [(0,36), (37,77), (78,113), (114,155), (156,193), (194,238), (239,274), (275,322), (323,358), (359,400)], 
            [(0,36), (37,82), (83,118), (119,160), (161,201), (202,249), (250,293), (294,335), (336,377), (378,416)]
        ],
        # Participant 5
        [
            [(0,54), (55,117), (118,170), (171,229), (230,283), (284,332), (333,390), (391,448), (449,504), (505,550)], 
            [(0,55), (56,103), (104,160), (161,215), (216,277), (278,335), (336,389), (390,452), (453,502), (503,556)]
        ],
        # Participant 6
        [
            [(0,48), (49,98), (99,149), (150,196), (197,242), (243,289), (290,338), (339,385), (386,432), (433,485)], 
            [(0,48), (49,96), (97,142), (143,189), (190,236), (237,286), (287,335), (336,386), (387,445)]
        ],
        # Participant 7
        [
            [(0,46), (47,90), (91,142), (143,188), (189,234), (235,281), (282,328), (329,382), (383,428), (429,480)], 
            [(0,42), (43,88), (89,134), (135,180), (181,225), (226,272), (273,317), (318,367), (368,411), (412,469)]
        ],
        # Participant 8
        [
            [(0,32), (33,69), (70,110), (111,151), (152,191), (192,232), (233,272), (273,314), (315,356), (357,402)], 
            [(0,34), (35,68), (69,107), (108,143), (144,180), (181,218), (219,256), (257,293), (294,330), (331,376)]
        ]
    ]

    activity1_indexes_per_healthy_C_DURING = [
        # Participant 1
        [
            [(0,81), (82,152), (153,224), (225,294), (295,373), (374,453), (454,526), (527,608), (609,695), (696,774), (775,865), (2277,2335), (2336,2400), (2401,2467), (2468,2532), (2533,2602), (2603,2674), (2675,2741), (2742,2809), (2810,2882), (2883,2951), (2952,3013), (3014,3085), (3086,3160), (4210,4267), (4268,4330), (4331,4398), (4399,4468), (4469,4535), (4536,4600), (4601,4667), (4668,4736), (4737,4804), (4805,4872), (4873,4944), (5957,6013), (6014,6082), (6083,6148), (6149,6215), (6216,6295), (6296,6356), (6357,6421), (6422,6486), (6487,6561), (6562,6626), (6627,6700), (7645,7702), (7703,7770), (7771,7836), (7837,7897), (7898,7973), (7974,8038), (8039,8103), (8104,8169), (8170,8238), (8239,8310)], 
            [(1170,1237), (1238,1296), (1297,1366), (1367,1430), (1431,1502), (1503,1580), (1581,1649), (1650,1720), (1721,1788), (1789,1860), (1861,1936), (1937,2005), (3293,3353), (3354,3425), (3426,3485), (3486,3557), (3558,3635), (3636,3693), (3694,3762), (3763,3835), (3836,3898), (3899,3967), (3968,4040), (5030,5093), (5094,5162), (5163,5228), (5229,5303), (5304,5371), (5372,5453), (5454,5525), (5526,5587), (5588,5669), (5670,5735), (5736,5810), (6818,6877), (6878,6944), (6945,7013), (7014,7076), (7077,7147), (7148,7220), (7221,7292), (7293,7354), (7355,7422), (7423,7505), (8415,8483), (8484,8550), (8551,8616), (8617,8682), (8683,8764), (8765,8827), (8828,8902), (8903,8966), (8967,9037), (9038,9110)]
        ],
        # Participant 2
        [
            [(0,90), (91,174), (175,279), (280,356), (357,460), (461,553), (554,640), (641,732), (733,820), (821,901), (902,1000), (1001,1086), (1087,1204), (1205,1318), (1319,1422), (1423,1532), (1523,1625), (3589,3684), (3685,3782), (3783,3880), (3881,3982), (3983,4087), (4088,4187), (4188,4285), (4286,4381), (4382,4487), (4488,4589), (4590,4680), (4681,4795), (5060,5159), (7493,7623), (7624,7765), (7766,7885), (7886,7995), (7996,8100), (8101,8219), (8220,8325), (8326,8440), (8441,8559), (8560,8680), (8681,8794), (8795,8890)], 
            [(1763,1851), (1852,1945), (1946,2034), (2035,2126), (2127,2220), (2221,2308), (2309,2394), (2395,2497), (2498,2585), (2586,2679), (2680,2768), (2769,2868), (2869,2970), (2971,3050), (5525,5632), (5633,5747), (5748,5862), (5863,5978), (5979,6092), (6093,6220), (6221,6335), (6336,6451), (6452,6550), (6620,6718), (6719,6825), (6826,6927), (6928,7029), (7030,7118)]
        ],
        # Participant 3
        [
            [(0,82), (83,181), (182,270), (271,359), (360,441), (442,528), (529,607), (608,701), (702,788), (789,878), (879,961), (962,1043), (1044,1131), (1132,1219), (1220,1298), (2530,2613), (2614,2707), (2708,2800), (2801,2891), (2892,2983), (2984,3075), (3076,3162), (3163,3252), (3253,3337), (3338,3428), (3429,3517), (3518,3607), (3608,3700), (3701,3785), (4879,4972), (4973,5064), (5065,5163), (5164,5253), (5254,5347), (5348,5435), (5436,5528), (5529,5616), (5617,5712), (5713,5808), (6920,7009), (7010,7099), (7100,7200), (7201,7291), (7292,7381), (7382,7470), (7471,7560), (7561,7638), (7639,7728)], 
            [(1325,1408), (1409,1497), (1498,1590), (1591,1671), (1672,1761), (1762,1837), (1838,1922), (1923,2003), (2004,2085), (2086,2171), (2172,2256), (2257,2344), (2345,2432), (2433,2515), (3800,3892), (3893,3986), (3987,4077), (4078,4164), (4165,4261), (4262,4350), (4351,4447), (4448,4529), (4530,4625), (4626,4715), (4716,4800), (5866,5961), (5962,6051), (6052,6135), (6136,6231), (6232,6330), (6331,6421), (6422,6519), (6520,6616), (6617,6710), (6711,6800)]
        ],
        # Participant 4
        [
            [(0,49), (50,97), (98,147), (148,195), (196,242), (243,289), (290,339), (340,387), (388,436), (437,484), (485,531), (532,577), (578,624), (625,671), (672,716), (717,762), (763,812), (813,861), (862,913), (914,973), (2195,2239), (2240,2283), (2284,2333), (2334,2381), (2382,2427), (2428,2477), (2478,2527), (2528,2578), (2579,2623), (2624,2671), (2672,2725), (2726,2770), (2771,2821), (2822,2870), (2871,2933), (5151,5196), (5197,5237), (5238,5281), (5282,5322), (5323,5371), (5372,5417), (5418,5464), (5465,5508), (5509,5553), (5554,5602), (5603,5648), (5649,5691), (5692,5736), (5737,5788), (5789,5832), (5833,5880), (6772,6811), (6812,6854), (6855,6897), (6898,6941), (6942,6986), (6987,7029), (7030,7075), (7076,7121),(7122,7167), (7168,7213), (7214,7259), (7260,7305), (7306,7362), (8375,8423), (8424,8471), (8472,8518), (8519,8564), (8565,8614), (8615,8657), (8658,8702), (8703,8750), (8751,8796)], 
            [(1167,1209), (1210,1256), (1257,1307), (1308,1352), (1353,1400), (1401,1448), (1449,1497), (1498,1545), (1546,1592), (1593,1637), (1638,1688), (1689,1737), (1738,1781), (1782,1835), (3177,3220), (3221,3268), (3269,3313), (3314,3362), (3363,3412), (3413,3459), (3460,3505), (3506,3552), (3553,3598), (3599,3643), (3644,3693), (3694,3739), (3740,3778), (3779,3826), (3827,3868), (3869,3915), (3916,3960), (3961,4008), (4009,4053), (4054,4101), (4102,4148), (4149,4205), (4206,4252), (4253,4295), (4296,4344), (4345,4390), (4391,4441), (4442,4493), (4494,4536), (4537,4582), (4583,4629), (4630,4690), (4691,4734), (4735,4792), (4793,4861), (6000,6052), (6053,6091), (6092,6136), (6137,6179), (6180,6224), (6225,6270), (6271,6316), (6317,6362), (6363,6403), (6404,6445), (6446,6490), (6491,6535), (6536,6583), (6584,6634), (7468,7516), (7517,7559), (7560,7608), (7609,7656), (7657,7705), (7706,7748), (7749,7801), (7802,7850), (7851,7895), (7896,7943), (7944,7993), (7994,8049), (8050,8100), (8101,8142), (8143,8190), (8191,8240)]
        ],
        # Participant 5
        [
            [(0,87), (88,169), (170,258), (259,331), (332,410), (411,491), (492,569), (570,660), (661,752), (753,838), (2337,2418), (2419,2509), (2510,2582), (2583,2673), (2674,2746), (2747,2814), (2815,2900), (2901,2981), (2982,3060), (3061,3149), (3150,3237), (3238,3320), (3321,3400), (3401,3484), (4969,5050), (5051,5122), (5123,5203), (5204,5280), (5281,5370), (5371,5444), (5445,5530), (5531,5603), (5604,5682), (5683,5766), (5767,5843), (5844,5939), (5940,6018), (6019,6110), (6111,6201), (6202,6291), (7410,7490), (7491,7581), (7582,7662), (7663,7759), (7760,7831), (7832,7917), (7918,7998), (7999,8084), (8085,8168), (8394,8478)], 
            [(1193,1278), (1279,1348), (1349,1429), (1430,1504),(1505,1578), (1579,1659), (1660,1735), (1736,1808), (1809,1895), (1896,1994), (1995,2078), (2079,2159), (2160,2237), (3662,3751), (3752,3843), (3844,3911), (3912,4001), (4002,4082), (4083,4167), (4168,4247), (4248,4319), (4320,4394), (4395,4486), (4487,4574), (4575,4646), (4647,4733), (4734,4823), (6450,6524), (6525,6607), (6608,6674), (6675,6757), (6758,6824), (6825,6909), (6910,6991), (6992,7070), (7071,7153), (7154,7230), (8612,8703), (8704,8782), (8783,8855), (8856,8943), (8944,9018), (9019,9102), (9103,9184), (9185,9267), (9268,9347), (9348,9422)]
        ],
        # Participant 6
        [
            [(0,66), (67,140), (141,226), (227,316), (317,397), (398,478), (479,559), (560,635), (636,724), (725,794), (1806,1876), (1877,1950), (1951,2025), (2026,2099), (2100,2174), (2175,2247), (2248,2322), (2323,2407), (2408,2485), (2486,2569), (3680,3748), (3749,3823), (3824,3899), (3900,3969), (3970,4049), (4050,4130), (4131,4205), (4206,4283), (4284,4360), (4361,4435), (5385,5462), (5463,5534), (5535,5606), (5607,5686), (5687,5760), (5761,5836), (5837,5912), (5913,5989), (5990,6067), (7254,7316), (7317,7387), (7388,7461), (7462,7532), (7533,7604), (7605,7680), (7681,7745), (7746,7820)], 
            [(977,1053), (1054,1127), (1128,1204), (1205,1277), (1278,1357), (1358,1435), (1436,1511), (1512,1591), (1592,1667), (2715,2781), (2782,2856), (2857,2930), (2931,3004), (3005,3079), (3080,3156), (3157,3229), (3230,3306), (3307,3382), (3383,3455), (4606,4674), (4675,4745), (4746,4821), (4822,4896), (4897,4972), (4973,5046), (5047,5122), (5123,5198), (5199,5273), (6184,6259), (6260,6335), (6336,6409), (6410,6489), (6490,6562), (6563,6638), (6639,6712), (6713,6789), (6790,6862), (6863,6935)]
        ],
        # Participant 7
        [
            [(0,65), (66,128), (129,188), (189,248), (249,312), (313,370), (371,435), (436,486), (487,549), (550,620), (621,677), (678,730), (731,793), (2062,2121), (2122,2182), (2183,2240), (2241,2297), (2298,2355), (2356,2416), (2417,2478), (2479,2542), (2543,2605), (2606,2652), (2653,2714), (2715,2779), (2780,2838), (2839,2900), (2901,2958), (2959,3024), (3025,3095), (3096,3160), (3161,3220), (4598,4652), (4653,4718), (4719,4778), (4779,4851), (4852,4904), (4905,4975), (4976,5039), (5040,5106), (5107,5173), (5174,5237), (5238,5293), (5294,5360), (5361,5432), (5433,5494), (5495,5551), (5552,5612), (5613,5687), (5688,5753), (5754,5813), (7282,7340), (7341,7403), (7404,7463), (7464,7527), (7528,7591), (7592,7648), (7649,7708), (7709,7774), (7775,7835), (7836,7890)], 
            [(1158,1212), (1213,1271), (1272,1328), (1329,1387), (1388,1445), (1446,1501), (1502,1561), (1562,1623), (1624,1685), (1686,1745), (1746,1808), (1809,1869), (3439,3499), (3500,3557), (3558,3614), (3615,3671), (3672,3730), (3731,3793), (3794,3851), (3852,3911), (3912,3974), (3975,4035), (4036,4107), (4108,4162), (4163,4229), (4230,4285), (4286,4353), (4354,4412), (5958,6014), (6015,6071), (6072,6131), (6132,6188), (6189,6246), (6247,6309), (6310,6370), (6371,6430), (6431,6489), (6490,6549), (6550,6608), (6609,6669), (6670,6726), (6727,6786), (6787,6848), (6849,6916), (6917,6973), (6974,7038), (7039,7101), (8033,8103), (8104,8170), (8171,8235), (8236,8306), (8307,8359), (8360,8426), (8427,8488), (8489,8549), (8550,8612), (8613,8675), (8676,8736), (8737,8796), (8797,8856)]
        ],
        # Participant 8
        [
            [(0,34), (35,77), (78,119), (120,165), (166,209), (210,251), (252,297), (298,340), (341,385), (386,431), (432,477), (1124,1165), (1166,1207), (1208,1250), (1251,1294), (1295,1338), (1339,1383), (1384,1428), (1429,1470), (1471,1514), (1515,1559), (1560,1611), (2264,2309), (2310,2351), (2352,2395), (2396,2440), (2441,2483), (2484,2526), (2527,2570), (2571,2612), (2613,2655), (2656,2699), (2700,2748), (3409,3450), (3451,3493), (3494,3535), (3536,3578), (3579,3623), (3624,3667), (3668,3711), (3712,3754), (3755,3799), (3800,3843), (3844,3896), (4570,4614), (4615,4657), (4658,4700), (4701,4743), (4744,4787), (4788,4831), (4832,4875), (4876,4921), (4922,4964), (4965,5008), (5009,5055), (5690,5734), (5735,5778), (5779,5822), (5823,5865), (5866,5908), (5909,5953), (5954,5996), (5997,6040), (6041,6086), (6087,6128), (6129,6178), (6818,6862), (6863,6905), (6906,6949), (6950,6993), (6994,7037), (7038,7083), (7084,7127), (7128,7172), (7173,7219), (7220,7267), (7268,7315)], 
            [(563,604), (605,643), (644,684), (685,724), (725,766), (767,807), (808,851), (852,893), (894,934), (935,976), (977,1026), (1695,1737), (1738,1780), (1781,1821), (1822,1865), (1866,1906), (1907,1949), (1950,1992), (1993,2038), (2039,2081), (2082,2125), (2126,2174), (2839,2880), (2881,2921), (2922,2963), (2964,3005), (3006,3047), (3048,3088), (3089,3130), (3131,3172), (3173,3212), (3213,3255), (3256,3301), (4005,4042), (4043,4085), (4086,4128), (4129,4170), (4171,4215), (4216,4256), (4257,4298), (4299,4341), (4342,4384), (4385,4426), (4427,4472), (5141,5181), (5182,5223), (5224,5266), (5267,5310), (5311,5356), (5357,5398), (5399,5441), (5442,5485), (5486,5532), (5533,5576), (5577,5622), (6257,6298), (6299,6340), (6341,6383), (6384,6427), (6428,6469), (6470,6512), (6513,6556), (6557,6600), (6601,6642), (6643,6686), (6687,6735), (7395,7435), (7436,7480), (7481,7524), (7525,7568), (7569,7613), (7614,7660), (7661,7707), (7708,7750), (7751,7793), (7794,7837), (7838,7885)]
        ]
    ]

    activity1_indexes_per_healthy_C_POS = [
        # Participant 1
        [
            [(0,63), (64,128), (129,182), (183,251), (252,314), (315,380), (381,442), (443,505), (506,577)], 
            [(0,55), (56,111), (112,175), (176,231), (232,294), (295,354), (355,411), (412,470), (471,537), (538,598)]
        ],
        # Participant 2
        [
            [(0,76), (77,155), (156,231), (232,308), (309,385), (386,458), (459,527), (528,597), (598,655)], 
            [(0,60), (61,131), (132,197), (198,266), (267,332), (333,397), (398,463), (464,528), (529,610), (611,664)]
        ],
        # Participant 3
        [
            [(0,77), (78,153), (154,229), (230,310), (311,383), (384,459), (460,548), (549,630), (631,706), (707,815)], 
            [(0,74), (75,153), (154,225), (226,300), (301,376), (377,449), (450,525), (526,599), (600,672), (673,758)]
        ],
        # Participant 4
        [
            [(0,53), (54,106), (107,161), (162,214), (215,269), (270,322), (323,368), (369,423), (424,473), (474,522)], 
            [(0,41), (42,105), (106,156), (157,208), (209,268), (269,327), (328,382), (383,435), (436,493), (494,537)]
        ],
        # Participant 5
        [
            [(0,59), (60,118), (119,188), (189,259), (260,327), (328,400), (401,464), (465,528), (529,588), (589,656)], 
            [(0,73), (74,139), (140,207), (208,269), (270,340), (341,404), (405,468), (469,530), (531,590), (591,648)]
        ],
        # Participant 6
        [
            [(0,61), (62,121), (122,182), (183,246), (247,307), (308,371), (372,429), (430,490), (491,551), (552,616)], 
            [(0,56), (57,122), (123,183), (184,246), (247,309), (310,372), (373,436), (437,500), (501,561), (562,621)]
        ],
        # Participant 7
        [
            [(0,37), (38,85), (86,133), (134,188), (189,232), (233,285), (286,328), (329,379), (380,425), (426,476)], 
            [(0,43), (44,89), (90,135), (136,183), (184,228), (229,272), (273,318), (319,365), (366,411)]
        ],
        # Participant 8
        [
            [(0,36), (37,74), (75,116), (117,157), (158,201), (202,244), (245,286), (287,330), (331,372), (373,418), (419,465)], 
            [(0,37), (38,79), (80,121), (122,161), (162,204), (205,246), (247,289), (290,330), (331,373), (374,415), (416,462)]
        ]
    ]

    # --------------------------- ACTIVITY 2 ---------------------------

    activity2_indexes_per_healthy_A_PRE = [
        # Participant 1
        [(0,212), (213,434), (435,642), (643,860), (861,1059), (1060,1274), (1275,1491), (1492,1709), (1710,1924), (1925,2156)],

        # Participant 2
        [(0,81), (82,211), (212,414), (415,615), (616,797), (798,990), (991,1173), (1174,1360), (1361,1549), (1550,1737), (1738,1926)],

        # Participant 3
        [(0,159), (160,334), (335,507), (508,699), (700,884), (885,1079), (1080,1268), (1269,1464), (1465,1646), (1647,1857)],

        # Participant 4
        [(0,244), (245,484), (485,725), (726,974), (975,1238), (1239,1515), (1516,1762), (1763,2017)],
        
        # Participant 5
        [(0,167), (168,342), (343,517), (518,714), (715,908), (909,1102), (1103,1288), (1289,1471), (1472,1650)],

        # Participant 6
        [(0,205), (206,392), (393,585), (586,793), (794,1005), (1006,1211), (1212,1411), (1412,1602), (1603,1792), (1793,2012)],

        # Participant 7
        [(223,452), (453,670), (671,881), (882,1091), (1092,1307), (1308,1523), (1524,1721), (1722,1917), (1918,2131)],

        # Participant 8
        [(0,239), (240,456), (457,709), (710,924), (925,1162), (1163,1389), (1390,1615), (1616,1838), (1839,2056), (2057,2275)]
    ]

    activity2_indexes_per_healthy_A_POS = [
        # Participant 1
        [(0,260), (261,540), (541,829), (830,1105), (1106,1386), (1387,1658), (1659,1929)],

        # Participant 2
        [(236,502), (503,743), (744,993), (994,1261), (1262,1480), (1481,1727), (1728,1970), (1971,2169)],

        # Participant 3
        [(0,194), (195,386), (387,578), (579,782), (783,986), (987,1181), (1182,1378), (1379,1577), (1578,1768), (1769,1995)],

        # Participant 4
        [(0,275), (276,555), (556,873), (874,1205), (1206,1529), (1530,1845), (1846,2125)],

        # Participant 5
        [(0,209), (210,451), (452,677), (678,921), (922,1171), (1172,1403), (1404,1622), (1623,1851), (1852,2099), (2100,2280)],

        # Participant 6
        [(0,187), (188,390), (391,597), (598,809), (810,1015), (1016,1219), (1220,1423), (1424,1618), (1619,1824), (1825,2027), (2028,2226)],

        # Participant 7
        [(0,239), (240,462), (463,687), (688,927), (928,1150), (1151,1388), (1389,1630), (1631,1853), (1854,2062), (2063,2283)],

        # Participant 8
        [(0,238), (239,470), (471,709), (710,950), (951,1193), (1194,1435), (1436,1679), (1680,1915), (1916,2154), (2155,2409)]
    ]

    activity2_indexes_per_healthy_B_PRE = [
        # Participant 1
        [(0,224), (225,456), (457,673), (674,890), (891,1095), (1096,1300), (1301,1496), (1497,1695), (1696,1898), (1899,2121)],

        # Participant 2
        [(0,250), (251,543), (544,754), (755,1007), (1008,1251), (1252,1493), (1494,1719), (1720,1945), (1946,2175)],

        # Participant 3
        [(0,237), (238,460), (461,656), (657,841), (842,1021), (1022,1203), (1204,1388), (1389,1574), (1575,1759), (1760,1955)],

        # Participant 4
        [(0,271), (272,538), (539,812), (813,1079), (1080,1354), (1355,1627), (1628,1889), (1890,2152)],

        # Participant 5
        [(0,239), (240,473), (474,703), (704,930), (931,1153), (1154,1367), (1368,1580), (1581,1789), (1790,1977), (1978,2165)],

        # Participant 6
        [(0,164), (165,341), (342,533), (534,728), (729,921), (922,1112), (1113,1290), (1291,1477), (1478,1674), (1675,1880)],

        # Participant 7
        [(0,171), (172,351), (352,540), (541,719), (720,908), (909,1095), (1096,1274), (1275,1454), (1455,1621), (1622,1805)],

        # Participant 8
        [(0,203), (204,426), (427,648), (649,870), (871,1097), (1098,1314), (1315,1548), (1549,1780), (1781,2015), (2016,2236)]
    ]

    activity2_indexes_per_healthy_B_POS = [
        # Participant 1
        [(0,278), (279,553), (554,804), (805,1067), (1068,1341), (1342,1613), (1614,1878), (1879,2147)],

        # Participant 2
        [(0,245), (246,537), (538,827), (828,1103), (1104,1395), (1396,1655), (1656,1935), (1936,2180)],

        # Participant 3
        [(0,185), (186,361), (362,544), (545,732), (733,921), (922,1101), (1102,1273), (1274,1445), (1446,1621), (1622,1816)],

        # Participant 4
        [(0,277), (278,547), (548,831), (832,1100), (1101,1372), (1373,1653), (1654,1935), (1936,2209), (2210,2468), (2469,2725)],

        # Participant 5
        [(0,215), (216,451), (452,650), (651,860), (861,1101), (1102,1313), (1314,1523), (1524,1748), (1749,1947), (1948,2171)],

        # Participant 6
        [(0,160), (161,345), (346,534), (535,732), (733,925), (926,1122), (1123,1301), (1302,1489), (1490,1656), (1657,1855)],

        # Participant 7
        [(0,194), (195,421), (422,644), (645,873), (874,1119), (1120,1358), (1359,1583), (1584,1807), (1808,2030), (2031,2262)],

        # Participant 8
        [(229,477), (478,724), (725,973), (974,1214), (1215,1448), (1449,1675), (1676,1907), (1908,2134), (2135,2355), (2356,2572)]
    ]

    activity2_indexes_per_healthy_C_PRE = [
        # Participant 1
        [(0,267), (268,509), (510,737), (738,977), (978,1207), (1208,1448), (1449,1685), (1686,1911), (1912,2148)],

        # Participant 2
        [(0,210), (211,420), (421,631), (632,832), (833,1047), (1048,1255), (1256,1481), (1482,1699), (1700,1910), (1911,2118)],

        # Participant 3
        [(0,317), (318,573), (574,808), (809,1029), (1030,1251), (1252,1480), (1481,1700), (1701,1912), (1913,2121)],

        # Participant 4
        [(0,224), (225,431), (432,628), (629,834), (835,1034), (1035,1241), (1242,1442), (1443,1636), (1637,1829), (1830,2020)],

        # Participant 5
        [(0,172), (173,348), (349,527), (528,726), (727,911), (912,1092), (1093,1269), (1270,1445), (1446,1608)],

        # Participant 6
        [(0,172), (173,352), (353,542), (543,723), (724,915), (916,1108), (1109,1298), (1299,1486), (1487,1683)],

        # Participant 7
        [(0,186), (187,407), (408,629), (630,842), (843,1056), (1057,1264), (1265,1468), (1469,1668), (1669,1874), (1875,2096)],

        # Participant 8
        [(0,184), (185,397), (398,604), (605,816), (817,1028), (1029,1248), (1249,1462), (1463,1674), (1675,1887), (1888,2099)],
    ]

    activity2_indexes_per_healthy_C_DURING = [
        # Participant 1
        [(0,352), (353,723), (724,1074), (1075,1541), (2170,2568), (2745,3139), (3140,3552), (3553,4018), (4577,4958), (4959,5367), (5368,6187), (6188,6740), (6741,7312), (8300,8917)],

        # Participant 2
        [(0,351), (352,860), (861,1385), (1386,1948), (1949,2459), (2460,2878), (2879,3345), (3346,3915), (3916,4425), (4426,4928), (4929,5420), (5421,5939), (5940,6415), (6416,6842), (6843,7256)],

        # Participant 3
        [(996,1296), (1297,1587), (1588,1857), (1858,2114), (2115,2365), (5266,5553), (5554,5847), (5848,6121), (6122,6387), (6388,6630), (7310,7620), (7661,7970), (7971,8254), (8255,8549)],

        # Participant 4
        [(0,226), (227,471), (472,713), (714,943), (944,1181), (1182,1413), (1414,1637), (1638,1892), (1893,2139), (2140,2384), (2385,2641), (2642,2883), (2884,3150), (3151,3401), (3402,3664), (3665,3923), (3924,4183), (4184,4448), (4449,4720), (4721,4959), (4960,5205), (5206,5455), (5456,5705), (5706,5974), (5975,6228), (6229,6496), (6497,6756), (6757,7025), (7026,7298)],

        # Participant 5
        [(0,253), (254,514), (515,800), (1143,1489), (1490,1805), (1806,2157), (2158,2515), (3329,3650), (3651,3976), (4621,4942), (6047,6344), (6660,7000), (7001,7275)],

        # Participant 6
        [(0,208), (209,422), (423,635), (636,856), (857,1088), (1089,1315), (1316,1554), (1555,1799), (1800,2036), (2037,2283), (2284,2512), (2513,2756), (2757,2994), (2995,3226), (3227,3469), (3470,3704), (3705,3944), (3945,4190), (4191,4437), (4438,4692), (4693,4941), (4942,5188), (5189,5424), (5425,5663), (5664,5893), (5894,6135), (6136,6372), (6373,6619), (6620,6865), (6866,7124), (7125,7376), (7377,7640)],

        # Participant 7
        [(0,271), (272,530), (531,772), (773,1044), (1045,1282), (1283,1519), (1520,1754), (1755,1988), (1989,2221), (2222,2465), (2566,2718), (2719,2979), (2980,3226), (3227,3461), (3462,3708), (3709,3947), (3948,4189), (4190,4431), (4432,4669), (4670,4906), (4907,5150), (5151,5399), (5400,5654), (5655,5922), (5923,6182), (6183,6430), (6431,6691), (6692,6955), (6956,7214), (7215,7475)],

        # Participant 8
        [(0,204), (205,434), (435,663), (664,908), (909,1155), (1156,1389), (1390,1620), (1621,1860), (1861,2104), (2105,2344), (2345,2593), (2594,2835), (2836,3084), (3085,3336), (3337,3586), (3587,3848), (3849,4091), (4371,4641), (4642,4910), (4911,5179), (5180,5439), (5699,5965), (6215,6476), (6477,6731), (6732,6990), (6991,7276)],
    ]

    activity2_indexes_per_healthy_C_POS = [
        # Participant 1
        [(0,272), (273,539), (540,813), (814,1088), (1089,1364), (1365,1638), (1639,1919), (1920,2193), (2194,2478), (2479,2774)],

        # Participant 2
        [(0,200), (201,410), (411,621), (622,822), (823,1037), (1038,1245), (1246,1471), (1472,1689), (1690,1900), (1901,2118)],

        # Participant 3
        [(0,209), (210,399), (400,596), (597,798), (799,994), (995,1186), (1187,1381), (1382,1569), (1570,1760), (1761,1942)],

        # Participant 4
        [(0,291), (292,560), (561,849), (850,1141), (1142,1425), (1426,1706), (1707,1975), (1976,2269), (2270,2550), (2551,2850)],

        # Participant 5
        [(0,162), (163,354), (355,567), (568,784), (785,981), (982,1182), (1183,1387), (1388,1591), (1592,1792), (1793,1993)],

        # Participant 6
        [(0,204), (205,410), (411,639), (640,870), (871,1100), (1101,1330), (1331,1562), (1563,1794), (1795,2018), (2019,2238)],

        # Participant 7
        [(0,220), (221,465), (466,704), (705,927), (928,1154), (1155,1393), (1394,1625), (1626,1854), (1855,2068), (2069,2267)],

        # Participant 8
        [(0,221), (222,447), (448,673), (674,892), (893,1114), (1115,1335), (1336,1545), (1546,1779), (1780,1992), (1993,2238)],
    ]

    # --------------------------- ACTIVITY 32 ---------------------------

    activity3_indexes_per_healthy_A_PRE = [
        # Participant 1
        [
            [(0,210), (211,485), (486,755), (756,1017), (1018,1274), (1275,1525), (1526,1800), (1801,2117), (2118,2412), (2413,2670)],
            [(0,220), (221,502), (503,765), (766,1042), (1043,1300), (1301,1557), (1558,1825), (1826,2075), (2076,2327), (2328,2608)]
        ],
        # Participant 2
        [
            [(0,170), (171,424), (425,655), (656,906), (907,1153), (1154,1393), (1394,1615), (1616,1861), (1862,2044), (2045,2220)],
            [(0,190), (191,396), (395,580), (581,804), (805,1013), (1014,1223), (1224,1406), (1407,1600), (1601,1780), (1781,1950)]
        ],
        # Participant 3
        [
            [(0,303), (304,620), (621,880), (881,1140), (1141,1433), (1434,1705), (1706,1966), (1967,2246), (2247,2471)], 
            [(0,268), (269,534), (535,800), (801,1042), (1043,1307), (1308,1564), (1565,1829), (1830,2065), (2066,2302), (2303,2551)]
        ],
        # Participant 4
        [
            [(0,403), (404,871), (872,1301), (1302,1748), (1749,2178), (2179,2576), (2577,3025), (3026,3445)], 
            [(0,338), (339,698), (699,1100), (1101,1528), (1529,1882), (1883,2270), (2271,2649), (2650,2988), (2989,3361), (3362,3638)]
        ],
        # Participant 5
        [
            [(0,110), (111,225), (226,366), (367,504), (505,657), (658,808), (809,953), (954,1099), (1100,1245), (1246,1390)], 
            [(0,108), (109,229), (230,369), (370,493), (494,623), (624,769), (770,889), (890,1015), (1016,1131), (1132,1270)]
        ],
        # Participant 6
        [
            [(0,447), (448,878), (879,1286), (1287,1678), (1679,2045), (2046,2424), (2425,2786), (2787,3146), (3147,3453)], 
            [(0,247), (248,574), (575,925), (926,1300), (1301,1655), (1656,2033), (2034,2363), (2364,2640), (2641,2949), (2950,3245)]
        ],
        # Participant 7
        [
            [(0,182), (183,433), (434,656), (657,885), (886,1147), (1148,1371), (1372,1591), (1592,1784)], 
            [(0,148), (149,338), (339,530), (531,709), (710,901), (902,1082), (1083,1255), (1256,1428), (1429,1590), (1591,1756)]
        ],
        # Participant 8
        [
            [(0,464), (465,939), (940,1381), (1382,1820), (1821,2226), (2227,2564), (2565,2878), (2879,3194), (3195,3501), (3502,3830)], 
            [(12,402), (403,908), (909,1373), (1374,1779), (1780,2235), (2236,2639), (2640,3045), (3046,3419), (3420,3804)]
        ]
    ]

    activity3_indexes_per_healthy_A_POS = [
        # Participant 1
        [
            [(0,364), (365,665), (666,985), (986,1363), (1364,1780), (1781,2200), (2201,2617), (2618,2986), (2987,3359), (3360,3670)],
            [(0,255), (256,685), (686,1170), (1171,1720), (1721,2260), (2261,2740), (2741,3232), (3233,3695)]
        ],
        # Participant 2
        [
            [(0,335), (336,722), (723,1071), (1072,1390), (1391,1722), (1723,2066), (2067,2385), (2386,2699), (2700,2982), (2983,3240)],
            [(0,264), (265,537), (538,828), (829,1131), (1132,1426), (1427,1716), (1717,2027), (2028,2331), (2332,2613), (2614,2860)]
        ],
        # Participant 3
        [
            [(0,389), (390,760), (761,1090), (1091,1423), (1424,1717), (1718,2012), (2013,2310), (2311,2601), (2602,2898), (2899,3151)], 
            [(0,201), (202,481), (482,752), (753,1015), (1016,1252), (1253,1495), (1496,1706), (1707,1961), (1962,2172), (2173,2375), (2376,2563)]
        ],
        # Participant 4
        [
            [(0,400), (401,816), (817,1190), (1191,1573), (1574,1963), (1964,2380), (2381,2749), (2750,3139), (3140,3512)], 
            [(0,378), (379,732), (733,1106), (1107,1485), (1486,1834), (1835,2257), (2258,2659), (2660,2985), (2986,3304)]
        ],
        # Participant 5
        [
            [(0,149), (150,327), (328,513), (514,688), (689,869), (870,1055), (1056,1239), (1240,1390), (1610,1741)], 
            [(0,129), (130,285), (286,465), (466,663), (664,852), (853,1019), (1020,1201), (1202,1368), (1369,1557), (1558,1711)]
        ],
        # Participant 6
        [
            [(0,539), (540,1182), (1183,1819), (1820,2432), (2433,3025), (3026,3692), (3693,4308), (4309,4936), (4937,5496), (5497,6003)], 
            [(0,463), (464,1117), (1118,1717), (1718,2354), (2355,2938), (2939,3550), (3551,4133), (4134,4770), (4771,5288), (5289,5733)]
        ],
        # Participant 7
        [
            [(0,169), (170,404), (405,629), (630,818), (819,1004), (1005,1184), (1185,1394), (1395,1565), (1566,1730), (1731,1919)], 
            [(0,139), (140,363), (364,596), (597,848), (849,1087), (1088,1303), (1304,1501), (1502,1686), (1687,1862), (1863,2037)]
        ],
        # Participant 8
        [
            [(0,574), (575,1118), (1119,1688), (1689,2213), (2214,2743), (2744,3233), (3234,3735), (3736,4581)], 
            [(0,779), (780,1499), (1500,2374), (2375,3166), (3167,3858), (3859,4563), (4564,5281), (5282,5964), (5965,6635), (6636,7260)]
        ]
    ]

    activity3_indexes_per_healthy_B_PRE = [
        # Participant 1
        [
            [(0,219), (220,439), (440,638), (639,839), (840,1043), (1044,1244), (1245,1435), (1436,1643), (1644,1864), (1865,2070)],
            [(0,197), (198,447), (448,702), (703,937), (938,1180), (1181,1400), (1401,1625), (1626,1869), (1870,2090)]
        ],
        # Participant 2
        [
            [(0,117), (118,241), (242,375), (376,517), (518,663), (664,795), (796,938), (939,1068), (1069,1200)],
            [(0,109), (110,240), (241,397), (398,550), (551,691), (692,817), (818,967), (968,1090), (1091,1245), (1246,1366)]
        ],
        # Participant 3
        [
            [(0,292), (293,583), (584,852), (853,1108), (1109,1391), (1392,1632), (1633,1881), (1882,2120), (2121,2353), (2354,2595)], 
            [(0,220), (221,470), (471,721), (722,993), (994,1235), (1236,1478), (1479,1734), (1735,1953), (1954,2179), (2180,2415)]
        ],
        # Participant 4
        [
            [(0,340), (341,739), (740,1152), (1153,1542), (1543,1930), (1931,2253), (2254,2611), (2612,2925), (2926,3256), (3257,3575)], 
            [(0,308), (309,632), (633,986), (987,1296), (1297,1628), (1629,1922), (1923,2233), (2234,2522), (2523,2834), (2835,3155)]
        ],
        # Participant 5
        [
            [(0,115), (116,316), (317,491), (492,650), (651,794), (795,949), (950,1106), (1107,1266), (1267,1429), (1430,1572)], 
            [(0,161), (162,336), (337,502), (503,681), (682,862), (863,1065), (1066,1239), (1240,1428), (1429,1605), (1606,1749)]
        ],
        # Participant 6
        [
            [(0,202), (203,471), (472,696), (697,935), (936,1159), (1160,1389), (1390,1613), (1614,1850), (1851,2074), (2075,2280)], 
            [(0,175), (176,380), (381,594), (595,799), (800,996), (997,1217), (1218,1390), (1391,1583), (1584,1760), (1761,1925)]
        ],
        # Participant 7
        [
            [(0,195), (196,536), (537,893), (894,1224), (1225,1535), (1536,1801), (1802,2078), (2079,2336), (2337,2579), (2580,2792)], 
            [(0,193), (194,445), (446,695), (696,932), (933,1179), (1180,1445), (1446,1722), (1723,1968), (1969,2199), (2200,2413)]
        ],
        # Participant 8
        [
            [(0,915), (916,1684), (1685,2420), (2425,3131), (3132,3819), (3820,4377), (4378,5000), (5001,5547), (5548,6165), (6166,6875)], 
            [(0,689), (690,1431), (1432,2191), (2192,2909), (2910,3537), (3538,4244), (4245,4920), (4921,5620), (5621,6202), (6203,6950)]
        ]
    ]

    activity3_indexes_per_healthy_B_POS = [
        # Participant 1
        [
            [(0,283), (284,618), (619,922), (923,1246), (1247,1617), (1618,1915), (1916,2208), (2209,2510), (2511,2822), (2823,3084)],
            [(0,242), (243,539), (540,843), (844,1204), (1205,1603), (1604,1990), (1991,2380), (2381,2770), (2771,3125)]
        ],
        # Participant 2
        [
            [(0,303), (304,610), (611,874), (875,1134), (1135,1383), (1384,1626), (1627,1870), (1871,2120), (2121,2336), (2337,2546)],
            [(0,192), (193,429), (430,659), (660,885), (886,1114), (1115,1325), (1326,1536), (1537,1749), (1750,1959), (1960,2161)]
        ],
        # Participant 3
        [
            [(0,303), (304,610), (611,874), (875,1134), (1135,1383), (1384,1626), (1627,1870), (1871,2120), (2121,2336), (2337,2550)], 
            [(0,207), (208,430), (431,660), (661,900), (901,1115), (1116,1334), (1335,1537), (1538,1750), (1751,1960), (1961,2173)]
        ],
        # Participant 4
        [
            [(0,330), (331,727), (728,1135), (1136,1515), (1516,1925), (1926,2288), (2289,2695), (2696,3100), (3101,3419), (3420,3700)], 
            [(0,309), (310,620), (621,963), (964,1303), (1304,1676), (1677,2091), (2092,2491), (2492,2835), (2836,3174), (3175,3498)]
        ],
        # Participant 5
        [
            [(0,173), (174,378), (379,555), (556,739), (740,907), (908,1077), (1078,1246), (1247,1411), (1412,1564), (1565,1715)], 
            [(0,177), (178,310), (311,490), (491,646), (647,832), (833,994), (995,1156), (1157,1324), (1325,1486), (1487,1640)]
        ],
        # Participant 6
        [
            [(0,428), (429,897), (898,1479), (1480,1976), (1977,2431), (2432,2946), (2947,3441), (3442,3842), (3843,4307), (4308,4728)], 
            [(0,362), (363,866), (867,1270), (1271,1792), (1793,2174), (2175,2588), (2589,2949), (2950,3363), (3364,3768), (3769,4138)]
        ],
        # Participant 7
        [
            [(0,210), (211,504), (505,760), (761,1010), (1011,1250), (1251,1542), (1543,1805), (1806,2071), (2072,2322), (2323,2516)], 
            [(0,187), (188,433), (434,715), (716,965), (966,1190), (1191,1414), (1415,1631), (1632,1830), (1831,2032), (2033,2235)]
        ],
        # Participant 8
        [
            [(0,766), (767,1322), (1323,2008), (2009,2614), (2615,3205), (3206,3729), (3730,4319), (4320,4867), (4868,5409)], 
            [(0,529), (530,1006), (1007,1527), (1528,2170), (2171,2753), (2754,3284), (3285,3819), (3820,4373), (4374,4950), (4951,5485)]
        ]
    ]

    activity3_indexes_per_healthy_C_PRE = [
        # Participant 1
        [
            [(0,240), (241,510), (511,778), (779,1020), (1021,1279), (1280,1548), (1549,1820), (1821,2074), (2075,2316)],
            [(0,199), (200,422), (423,671), (672,919), (920,1161), (1162,1430), (1431,1680), (1681,1926), (1927,2181), (2182,2422)]
        ],
        # Participant 2
        [
            [(0,191), (192,386), (387,600), (601,827), (828,1031), (1032,1232), (1233,1416), (1417,1619), (1620,1823), (1824,2005)],
            [(0,184), (185,395), (396,605), (606,787), (788,1010), (1011,1246), (1247,1462), (1463,1678), (1679,1898), (1899,2108)]
        ],
        # Participant 3
        [
            [(0,350), (351,666), (667,988), (989,1270), (1271,1533), (1534,1834), (1835,2154), (2155,2455), (2456,2749), (2750,2990)], 
            [(0,214), (215,406), (407,624), (625,859), (860,1095), (1096,1335), (1336,1556), (1557,1796), (1797,2012), (2013,2239)]
        ],
        # Participant 4
        [
            [(0,274), (275,624), (625,942), (943,1300), (1301,1602), (1603,1944), (1945,2294), (2295,2591), (2592,2883), (2884,3200)], 
            [(0,264), (265,529), (530,802), (803,1050), (1051,1280), (1281,1549), (1550,1839), (1840,2105), (2106,2330), (2331,2569)]
        ],
        # Participant 5
        [
            [(0,163), (164,318), (319,473), (474,636), (637,817), (818,994), (995,1163), (1164,1320), (1321,1466), (1467,1627)], 
            [(0,128), (129,279), (280,454), (455,639), (640,792), (793,965), (966,1130), (1131,1289), (1290,1449), (1450,1583)]
        ],
        # Participant 6
        [
            [(0,320), (321,750), (751,1160), (1161,1526), (1527,1844), (1845,2190), (2191,2548), (2549,2875), (2876,3204), (3205,3505)], 
            [(0,290), (291,601), (602,930), (931,1270), (1271,1620), (1621,1978), (1979,2300), (2301,2672), (2673,3056), (3057,3392)]
        ],
        # Participant 7
        [
            [(0,233), (234,547), (548,851), (852,1162), (1163,1507), (1508,1779), (1780,2076), (2077,2325), (2326,2587), (2588,2773)], 
            [(0,247), (248,520), (521,779), (780,1041), (1042,1307), (1308,1574), (1575,1835), (1836,2068), (2069,2293), (2294,2499)]
        ],
        # Participant 8
        [
            [(0,518), (519,1048), (1049,1576), (1577,2160), (2161,2722), (2723,3199), (3200,3655), (3656,4120), (4121,4646), (4647,5140)], 
            [(0,540), (541,1092), (1093,1577), (1578,2039), (2040,2496), (2497,2927), (2928,3356), (3357,3823), (3824,4283), (4284,4770), (4771,5292)]
        ]
    ]

    activity3_indexes_per_healthy_C_DURING = [
        # Participant 1
        [
            [(0,291), (292,548), (549,839), (840,1142), (1143,1426), (1427,1743), (1744,2046), (2047,2383), (2384,2699), (2700,3006)],
            [(0,262), (263,547), (548,824), (825,1106), (1107,1403), (1404,1706), (1707,2008), (2009,2312), (2313,2610), (2611,2920)]
        ],
        # Participant 2
        [
            [(0,313), (314,739), (740,1267), (1268,1759), (1760,2161), (2162,2671), (2672,3206), (3207,3570), (6689,7128), (7129,7517), (7518,7838), (7839,8125), (8126,8523)],
            [(3575,3920), (3921,4484), (4485,4913), (4914,5331), (5332,5777), (5778,6200), (6201,6490), (8620,9033), (9034,9400)]
        ],
        # Participant 3
        [
            [(0,252), (253,464), (465,688), (689,908), (909,1107), (1108,1329), (1330,1598), (1599,1921), (3928,4113), (4114,4338), (4339,4559), (4560,4777), (4778,4974), (4975,5218), (5219,5430), (5431,5656), (5657,5868), (5869,6135), (8455,8650), (8651,8840), (8841,9026), (9027,9254)], 
            [(2048,2342), (2343,2585), (2586,2843), (2844,3113), (3114,3346), (3347,3577), (3578,3791), (6218,6413), (6414,6613), (6614,6819), (6820,7006), (7007,7199), (7200,7379), (7380,7562), (7563,7750), (7751,7952), (7953,8188), (8189,8394)]
        ],
        # Participant 4
        [
            [(0,205), (206,487), (488,822), (823,1174), (1175,1562), (1563,1982), (1983,2344), (4190,4520), (4521,4860), (4861,5230), (5231,5596)], 
            [(2482,2816), (2817,3131), (3132,3469), (3470,3808), (3809,4075), (5723,6041), (6042,6358), (6359,6670), (6671,6988), (6989,7331), (7332,7680)]
        ],
        # Participant 5
        [
            [(0,150), (151,290), (291,511), (512,674), (675,958), (959,1220), (1221,1449), (1450,1702), (1703,1923), (1924,2146), (2147,2466), (2467,2661), (5020,5247), (5248,5437), (5438,5658), (5659,5933), (5934,6150), (6151,6371), (6372,6586), (6868,7094), (7095,7324), (7325,7550)], 
            [(2684,2857), (2858,3075), (3076,3297), (3298,3504), (3505,3804), (3805,4051), (4052,4308), (4309,4522), (4523,4724), (4725,4950), (7939,8153), (8154,8376), (8377,8595)]
        ],
        # Participant 6
        [
            [(0,496), (497,1149), (1150,1720), (1721,2349), (2350,2888), (2889,3375), (3376,3895), (3896,4473), (4474,5060)], 
            [(5190,5666), (5667,6251), (6252,6742), (6743,7231), (7232,7780), (7781,8236), (8237,8748), (8749,9152)]
        ],
        # Participant 7
        [
            [(0,331), (332,590), (591,874), (875,1125), (1126,1362), (1363,1621), (1622,1869), (1870,2134), (2135,2372), (2373,2618), (2619,2879), (2880,3201), (3202,3528), (3529,3830), (3831,4130)], 
            [(4212,4447), (4448,4725), (4726,5013), (5014,5730), (5731,5302), (5303,5601), (5602,5892), (5893,6194), (6195,6468), (6469,6780), (6781,7112), (7113,7419), (7420,7683), (7684,7987)]
        ],
        # Participant 8
        [
            [(0,715), (716,1378), (1379,2008), (2009,2592), (2593,3102), (3103,3625), (3626,4085), (4086,4640), (4641,5127)], 
            [(5145,5626), (5627,6163), (6164,6623), (6624,7048), (7049,7576), (7577,8073), (8074,8642), (8643,9118), (9119,9584), (9585,10111), (10112,10674)]
        ]
    ]

    activity3_indexes_per_healthy_C_POS = [
        # Participant 1
        [
            [(0,291), (292,548), (549,839), (840,1142), (1143,1426), (1427,1743), (1744,2046), (2047,2383), (2384,2699), (2700,3006)],
            [(0,262), (263,547), (548,824), (825,1106), (1107,1403), (1404,1706), (1707,2008), (2009,2312), (2313,2610), (2611,2920)]
        ],
        # Participant 2
        [
            [(0,235), (236,493), (494,718), (719,916), (917,1141), (1142,1345), (1346,1589), (1590,1834), (1835,2061), (2062,2275)],
            [(0,188), (189,422), (423,629), (630,860), (861,1064), (1065,1268), (1269,1478), (1479,1681), (1682,1880), (1881,2065)]
        ],
        # Participant 3
        [
            [(0,333), (334,626), (627,945), (946,1255), (1256,1544), (1545,1825), (1826,2089), (2090,2394), (2395,2650), (2651,2937)], 
            [(0,234), (235,521), (522,815), (816,1091), (1092,1364), (1365,1596), (1597,1824), (1825,2074), (2075,2316), (2317,2519)]
        ],
        # Participant 4
        [
            [(0,346), (347,626), (627,936), (937,1265), (1266,1587), (1588,1905), (1906,2220), (2221,2522), (2523,2838), (2839,3130)], 
            [(0,288), (289,552), (553,828), (829,1121), (1122,1410), (1411,1686), (1687,1971), (1972,2261), (2262,2513), (2514,2760)]
        ],
        # Participant 5
        [
            [(0,195), (196,406), (407,601), (602,793), (794,990), (991,1182), (1183,1368), (1369,1540), (1541,1708), (1709,1890)], 
            [(0,145), (146,308), (309,480), (481,612), (613,776), (777,943), (944,1106), (1107,1259), (1260,1416)]
        ],
        # Participant 6
        [
            [(0,504), (505,1090), (1091,1675), (1676,2267), (2268,2847), (2848,3417), (3418,3971), (3972,4504), (4505,5017), (5018,5522)], 
            [(0,450), (451,980), (981,1511), (1512,2010), (2011,2574), (2575,3140), (3141,3649), (3650,4196), (4197,4740)]
        ],
        # Participant 7
        [
            [(0,315), (316,671), (672,997), (998,1306), (1307,1595), (1596,1852), (1853,2124), (2125,2409), (2410,2618), (2619,2859)], 
            [(0,219), (220,461), (462,727), (728,975), (976,1238), (1239,1488), (1489,1707), (1708,1922), (1923,2110)]
        ],
        # Participant 8
        [
            [(0,672), (673,1224), (1225,1709), (1710,2157), (2158,2664), (2665,3180), (3181,3682), (3683,4121), (4122,4554)], 
            [(0,460), (461,856), (857,1320), (1321,1887), (1888,2507), (2508,3129), (3130,3680), (3681,4277), (4278,4750), (4751,5173)]
        ]
    ]

    # --------------------------- ACTIVITY 31 ---------------------------

    activity3_rest_indexes_per_healthy_A_PRE = [
        (0,626), # Participant 1
        (0,602), # Participant 2
        (0,627), # Participant 3
        (0,622), # Participant 4
        (0,628), # Participant 5
        (0,633), # Participant 6
        (0,616), # Participant 7
        (0,633)  # Participant 8
    ]

    activity3_rest_indexes_per_healthy_A_POS = [
        (0,704), # Participant 1
        (0,630), # Participant 2
        (0,647), # Participant 3
        (0,624), # Participant 4
        (0,637), # Participant 5
        (0,639), # Participant 6
        (0,634), # Participant 7
        (0,635)  # Participant 8
    ]

    activity3_rest_indexes_per_healthy_B_PRE = [
        (0,650), # Participant 1
        (0,670), # Participant 2
        (0,625), # Participant 3
        (0,640), # Participant 4
        (0,623), # Participant 5
        (0,633), # Participant 6
        (0,631), # Participant 7
        (0,638)  # Participant 8
    ]

    activity3_rest_indexes_per_healthy_B_POS = [
        (0,656), # Participant 1
        (0,636), # Participant 2
        (0,624), # Participant 3
        (0,629), # Participant 4
        (0,626), # Participant 5
        (0,624), # Participant 6
        (0,629), # Participant 7
        (0,604)  # Participant 8
    ]

    activity3_rest_indexes_per_healthy_C_PRE = [
        (0,638), # Participant 1
        (0,638), # Participant 2
        (0,633), # Participant 3
        (0,632), # Participant 4
        (0,631), # Participant 5
        (0,635), # Participant 6
        (0,661), # Participant 7
        (0,645) # Participant 8
    ]

    activity3_rest_indexes_per_healthy_C_DURING = [
        (0,1737), # Participant 1
        (0,1485), # Participant 2
        (0,1235), # Participant 3
        (0,1228), # Participant 4
        (0,2251), # Participant 5
        (0,1783), # Participant 6
        (0,1266), # Participant 7
        (0,1574) # Participant 8
    ]

    activity3_rest_indexes_per_healthy_C_POS = [
        (0,774), # Participant 1
        (0,638), # Participant 2
        (0,636), # Participant 3
        (0,631), # Participant 4
        (0,648), # Participant 5
        (0,631), # Participant 6
        (0,638), # Participant 7
        (0,633) # Participant 8
    ]

    # ---------------------------- PD INDEXES ----------------------------

    activity_indexes_per_PD_B_PRE = [
        # Participant 9
        [
            [(0,102), (103,216), (217,335), (336,455), (456,562), (563,662), (663,760), (761,876), (877,986), (987,1097), (1098,1220)],
            [(0,89), (90,202), (203,310), (311,425), (426,532), (533,644), (645,747), (748,851), (852,959), (960,1067)]
        ]
    ]

    activity_indexes_per_PD_B_POS = [
        # Participant 9
        [
            [(0,122), (123,240), (241,358), (359,477), (478,602), (603,718), (719,835), (836,941), (942,1061), (1062,1186)],
            [(0,116), (117,254), (255,383), (384,524), (525,656), (657,795), (796,927), (928,1056), (1057,1171), (1172,1293)]
        ]
    ]

    activity_indexes_per_PD_B_DURING = [
        # Participant 9
        [
            [(0,244), (245,430), (431,600), (601,778), (779,941), (942,1128), (1129,1406), (1407,1669), (1670,1931), (1932,2129)],
            [(0,187), (188,397), (398,581), (582,782), (783,970), (971,1139), (1140,1306), (1307,1491), (1492,1663), (1664,1854), (1960,2121), (2122,2285)]
        ]
    ]


    ########################################### DATA PROCESSING ###########################################

    df_healthy = create_healthy_data_packets(number_healthy)
    df_PD = create_PD_data_packets(number_PD, PD_activity)

    max_samples_per_participant, mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS = create_healthy_mean_curves_by_activity(df_healthy, number_healthy)

    mean_B_PRE, mean_B_POS, std_B_PRE, std_B_POS, mean_B_DURING, std_B_DURING = create_PD_mean_curves_by_activity(df_PD, number_PD, PD_activity)

    all_mean_1A_PRE, all_mean_1A_POS, all_mean_2A_PRE, all_mean_2A_POS, all_mean_31A_PRE, all_mean_31A_POS, all_mean_32A_PRE, all_mean_32A_POS, all_mean_1B_PRE, all_mean_1B_POS, all_mean_2B_PRE, all_mean_2B_POS, all_mean_31B_PRE, all_mean_31B_POS, all_mean_32B_PRE, all_mean_32B_POS, all_mean_1C_PRE, all_mean_1C_DURING, all_mean_1C_POS, all_mean_2C_PRE, all_mean_2C_DURING, all_mean_2C_POS, all_mean_31C_PRE, all_mean_31C_DURING, all_mean_31C_POS, all_mean_32C_PRE, all_mean_32C_DURING, all_mean_32C_POS, all_std_1A_PRE, all_std_1A_POS, all_std_2A_PRE, all_std_2A_POS, all_std_31A_PRE, all_std_31A_POS, all_std_32A_PRE, all_std_32A_POS, all_std_1B_PRE, all_std_1B_POS, all_std_2B_PRE, all_std_2B_POS, all_std_31B_PRE, all_std_31B_POS, all_std_32B_PRE, all_std_32B_POS, all_std_1C_PRE, all_std_1C_DURING, all_std_1C_POS, all_std_2C_PRE, all_std_2C_DURING, all_std_2C_POS, all_std_31C_PRE, all_std_31C_DURING, all_std_31C_POS, all_std_32C_PRE, all_std_32C_DURING, all_std_32C_POS = find_all_healthy_mean_curve(max_samples_per_participant, mean_1A_PRE, mean_1A_POS, mean_2A_PRE, mean_2A_POS, mean_31A_PRE, mean_31A_POS, mean_32A_PRE, mean_32A_POS, mean_1B_PRE, mean_1B_POS, mean_2B_PRE, mean_2B_POS, mean_31B_PRE, mean_31B_POS, mean_32B_PRE, mean_32B_POS, mean_1C_PRE, mean_1C_DURING, mean_1C_POS, mean_2C_PRE, mean_2C_DURING, mean_2C_POS, mean_31C_PRE, mean_31C_DURING, mean_31C_POS, mean_32C_PRE, mean_32C_DURING, mean_32C_POS)

    find_healthy_features("A", -1, all_mean_1A_PRE, all_std_1A_PRE, all_mean_1A_POS, all_std_1A_POS, all_mean_2A_PRE, all_std_2A_PRE, all_mean_2A_POS, all_std_2A_POS, all_mean_31A_PRE, all_std_31A_PRE, all_mean_31A_POS, all_std_31A_POS, all_mean_32A_PRE, all_std_32A_PRE, all_mean_32A_POS, all_std_32A_POS)

    find_healthy_features("B", -1, all_mean_1B_PRE, all_std_1B_PRE, all_mean_1B_POS, all_std_1B_POS, all_mean_2B_PRE, all_std_2B_PRE, all_mean_2B_POS, all_std_2B_POS, all_mean_31B_PRE, all_std_31B_PRE, all_mean_31B_POS, all_std_31B_POS, all_mean_32B_PRE, all_std_32B_PRE, all_mean_32B_POS, all_std_32B_POS)

    find_healthy_features("C", -1, all_mean_1C_PRE, all_std_1C_PRE, all_mean_1C_POS, all_std_1C_POS, all_mean_2C_PRE, all_std_2C_PRE, all_mean_2C_POS, all_std_2C_POS, all_mean_31C_PRE, all_std_31C_PRE, all_mean_31C_POS, all_std_31C_POS, all_mean_32C_PRE, all_std_32C_PRE, all_mean_32C_POS, all_std_32C_POS)

    find_PD_features(number_PD, PD_activity, mean_B_PRE, std_B_PRE, mean_B_POS, std_B_POS, mean_B_DURING, std_B_DURING)
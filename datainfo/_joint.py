from enum import Enum

''' Indices for joints.

FLIC and MPII joints are mixed.
'''


class JOINT(Enum):
    # FLIC
    L_Shoulder = 0
    R_Shoulder = 1
    L_Elbow = 2
    R_Elbow = 3
    L_Wrist = 4
    R_Wrist = 5
    L_Hip = 6
    R_Hip = 7
    L_Knee = 8
    R_Knee = 9
    L_Ankle = 10
    R_Ankle = 11

    '''

    L_Eye = 12
    R_Eye = 13
    L_Ear = 14
    R_Ear = 15
    M_Nose = 16

    M_Shoulder = 17
    M_Hip = 18
    M_Ear = 19
    M_Torso = 20
    M_LUpperArm = 21
    M_RUpperArm = 22
    M_LLowerArm = 23
    M_RLowerArm = 24
    M_LUpperLeg = 25
    M_RUpperLeg = 26
    M_LLowerLeg = 27
    M_RLowerLeg = 28
    '''

    # MPII
    M_Pelvis = 12
    M_Thorax = 13
    M_UpperNeck = 14
    M_HeadTop = 15

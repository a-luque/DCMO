import torch
import torch.nn as nn

CONTROLLERS = ['sport', 'aggressive', 'dynamic', 'balanced', 'comfort', 'conservative', 'defensive']


IDM_PROFILES = {

    "sport": {
        "IDM_T": 0.6,
        "IDM_S0": 0.8,
        "IDM_A": 4.0,
        "IDM_B": 4.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.08
    },

    "aggressive": {
        "IDM_T": 0.8,
        "IDM_S0": 1.2,
        "IDM_A": 3.5,
        "IDM_B": 4.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.12
    },

    "dynamic": {
        "IDM_T": 1.0,
        "IDM_S0": 1.8,
        "IDM_A": 3.0,
        "IDM_B": 3.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.18
    },

    "balanced": {
        "IDM_T": 1.3,
        "IDM_S0": 2.5,
        "IDM_A": 2.5,
        "IDM_B": 3.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.25
    },

    "comfort": {
        "IDM_T": 1.6,
        "IDM_S0": 3.2,
        "IDM_A": 2.0,
        "IDM_B": 2.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.35
    },

    "conservative": {
        "IDM_T": 2.0,
        "IDM_S0": 4.0,
        "IDM_A": 1.5,
        "IDM_B": 2.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.50
    },

    "defensive": {
        "IDM_T": 2.5,
        "IDM_S0": 5.0,
        "IDM_A": 1.0,
        "IDM_B": 1.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.70
    },
}

class IDM(nn.Module):
    def __init__(self, idm_name):
        super().__init__()
        self.IDM_V0 = 18
        self.idm_dict = IDM_PROFILES[idm_name]
        self.IDM_T = self.idm_dict["IDM_T"]
        self.IDM_S0 = self.idm_dict["IDM_S0"]
        self.IDM_A = self.idm_dict["IDM_A"]
        self.IDM_B = self.idm_dict["IDM_B"]
        self.IDM_DELTA = self.idm_dict["IDM_DELTA"]
        self.IDM_MAX_BRAKE = self.idm_dict["MAX_BRAKE"]
        self.ACTUATOR_TAU = self.idm_dict["ACTUATOR_TAU"]
        
    def forward(self, x):
        ve, vl, gap = x[:,0], x[:,1], x[:,2]
        # No leader
        idm_accel_no_leader = self.IDM_A * (1 - (ve / self.IDM_V0) ** self.IDM_DELTA)
        idm_accel_no_leader = torch.max(torch.min(idm_accel_no_leader, torch.FloatTensor([self.IDM_A]).expand_as(idm_accel_no_leader)), torch.FloatTensor([-self.IDM_MAX_BRAKE]).expand_as(idm_accel_no_leader))

        # Leader
        gap         = torch.max(gap, torch.FloatTensor([0.1]).expand_as(gap))                          # avoid division by zero
        delta_v     = ve - vl
        s_star      = self.IDM_S0 + torch.max(torch.FloatTensor([0]).expand_as(ve), ve * self.IDM_T + (ve * delta_v) / (2 * (self.IDM_A * self.IDM_B) ** 0.5))
        idm_accel_leader   = self.IDM_A * (1 - (ve / self.IDM_V0) ** self.IDM_DELTA - (s_star / gap) ** 2)

        idm_accel_leader = torch.max(torch.min(idm_accel_leader, torch.FloatTensor([self.IDM_A]).expand_as(idm_accel_leader)), torch.FloatTensor([-self.IDM_MAX_BRAKE]).expand_as(idm_accel_leader))
                
        # Combine
        res = torch.where(gap > 55, idm_accel_no_leader, idm_accel_leader).unsqueeze(1) 
        return res

class BasicMOE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [
                IDM(c) for c in CONTROLLERS
            ]
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(3, len(CONTROLLERS))
        )
        self.topk = 1

    def forward(self, x):
        # x shape: (batch, feature_in)
        expert_logits = self.gate(x[:,1:4]) # shape (batch, expert_number)
        expert_weight = torch.softmax(expert_logits, dim=1) # shape (batch, expert_number)

        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ] # each element shape (batch, )
        # concat (batch, expert_number, feature_out)
        expert_output = torch.concat(expert_out_list, dim=1)  
        expert_weight = expert_weight.unsqueeze(1) # (batch, 1, expert_number)
        output = expert_weight @ expert_output # (batch, 1, feature_out)
        return output.squeeze()

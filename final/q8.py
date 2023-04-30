P_B = 0.06
P_E = 0.02
P_A_given_B_E = 0.99
P_A_given_notB_E = 0.15
P_A_given_B_notE = 0.95
P_A_given_notB_notE = 0.0001

P_notA_given_B_notE = 1 - P_A_given_B_notE
P_notB = 1 - P_B
P_notE = 1 - P_E

P_A = (P_A_given_B_E * P_B * P_E +
       P_A_given_B_notE * P_B * P_notE +
       P_A_given_notB_E * P_notB * P_E +
       P_A_given_notB_notE * P_notB * P_notE)

P_B_notE_given_notA = (P_notA_given_B_notE * P_B * P_notE) / (1 - P_A)

print(f"Probability of Burglary and no Earthquake given no Alarm: {P_B_notE_given_notA}")
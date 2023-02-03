import random
import Func

import numpy as np
import math
import copy
class Game(object):
    def __init__(self):
        self.Iteration=1000
        self.AttackerTarget =800
        self.Belief_thr =200
        self.Sysnum = [10,10,10,10,10]
        
        self.Utility = [random.randint(1,20), random.randint(1,20), random.randint(1,20), random.randint(1,20), random.randint(1,20)]
        self.Fake_Utility = [random.randint(1,20), random.randint(1,20), random.randint(1,20), random.randint(1,20), random.randint(1,20)]
        self.Budget = 500
        self.Belief_list=[[],[]]
        #self.flag=0
        self.Obfuscate_para = 0.2
        self.Offline_para = 0.3
        self.Honepot_para = 0.5
        self.Attack_para = [0.2, 1]
        self.Epsilon = 0.3
        self.ori_beliefa =60
        self.ori_beliefd =0.5
        self.Feasible_matrix = [[1, 0, 1,0, 1], [0, 1, 1, 1, 0], [0, 1, 0,1, 1], [1, 0, 1, 0,1], [1, 0, 1, 1, 0]]
        #self.Feasible_matrix = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        self.RewardM, self.CostM,self.BeliefM =self.RewardM_compute()

    def exp_dff(self, fake_utility_, Dff_epsilon):
            probability0 = []

            probability1 = []
            max_u = max(fake_utility_)
            for score in fake_utility_:
                probability0.append(np.exp((Dff_epsilon * score) / (2 * max_u)))
            sum = np.sum(probability0)
            # 归一化处理
            for i in range(len(probability0)):
                probability1.append(probability0[i] / sum)

            return probability1

    def feasilbe_constrain(self, M):
            k = []
            for i in range(len(self.Utility)):
                constrain_list = []
                for j in range(len(self.Fake_Utility)):
                    if M[i][j]:
                        constrain_list.append(self.Fake_Utility[j])

                temp1 = self.exp_dff(constrain_list, self.Epsilon)

                temp2 = []
                item = 0
                for j in range(len(self.Fake_Utility)):

                    if M[i][j]:
                        temp2.append(temp1[item])
                        item += 1
                    else:
                        temp2.append(0)
                k.append(temp2)

            return k

    def RewardM_compute(self):

            prob = self.feasilbe_constrain(self.Feasible_matrix)
            b_ = 1 / self.Epsilon
            r = [
                [[0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]],
                 [0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]]],
                [[0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]],
                 [0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]]]
            ]
            b = [
                [[0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]],
                 [0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]]],
                [[0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]],
                 [0, [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]]]
            ]
            cost = [0, 0]
            rd_half = 0
            rd_total = 0

            ra_half = [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]
            ra_total = [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]
            ba_half = [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]
            ba_total = [0 for x in range(len(self.Fake_Utility) + len(self.Utility))]
            for i in range(len(self.Utility)):
                rd_half -= 0.6 * self.Sysnum[i] * self.Utility[i]
                # rd_total[i] = self.Sysnum[i]*self.Utility[i]
                ra_half[i] = 0.6 * self.Sysnum[i] * self.Utility[i]
                ba_half[i] = 0.6 * self.Sysnum[i] * self.Utility[i]
                for j in range(len(self.Fake_Utility)):
                    rd_half += 0.4* self.Sysnum[i] * prob[i][j] * self.Utility[j]
                    ra_half[len(self.Utility) + j] -= 0.4 * self.Sysnum[i] * prob[i][j] * self.Utility[j]
                    ba_half[len(self.Utility) + j] += 0.4 * self.Sysnum[i] * prob[i][j] * self.Utility[j]
                    rd_total += (self.Sysnum[i] + 0.5 * b_) * prob[i][j] * self.Utility[j]
                    ra_total[len(self.Utility) + j] -= (self.Sysnum[i] + 0.5 * b_) * prob[i][j] * self.Fake_Utility[j]
                    ba_total[len(self.Utility) + j] += (self.Sysnum[i] + 0.5 * b_) * prob[i][j] * self.Fake_Utility[j]
                    cost[0] += 0.5 * self.Sysnum[i] * prob[i][j] * abs(
                        self.Utility[i] - self.Fake_Utility[j]) * self.Obfuscate_para
                    cost[1] += self.Sysnum[i] * prob[i][j] * abs(
                        self.Utility[i] - self.Fake_Utility[j]) * self.Obfuscate_para
                cost[1] += 0.5 * b_ * (self.Offline_para + self.Honepot_para) * self.Utility[i]
            for i in range(len(self.Fake_Utility) + len(self.Utility)):
                r[0][0][1][i] = ra_half[i] * self.Attack_para[0]
                r[1][0][1][i] = ra_total[i] * self.Attack_para[0]
                r[0][1][1][i] = ra_half[i] * self.Attack_para[1]
                r[1][1][1][i] = ra_total[i] * self.Attack_para[1]
                b[0][0][1][i] = ba_half[i] * self.Attack_para[0]
                b[1][0][1][i] = ba_total[i] * self.Attack_para[0]
                b[0][1][1][i] = ba_half[i] * self.Attack_para[1]
                b[1][1][1][i] = ba_total[i] * self.Attack_para[1]

            r[0][0][0] = rd_half * self.Attack_para[0]
            r[0][1][0] = rd_half * self.Attack_para[1]
            r[1][0][0] = rd_total * self.Attack_para[0]
            r[1][1][0] = rd_total * self.Attack_para[1]

            return r, cost,b
    def Min(self, a, b):
        if a < 0 or b < 0:
            return 0
        if a > b:
            return b
        else:
            return a
    def Belief_Update(self,belief,ra,belief2,attack,ra_before):

        prob = [[0.3, 0.7], [0.8, 0.2]]
        if len(belief) == 0:
            b=copy.deepcopy(belief2)

            sum = 0
            for i in range(len(b)):
                sum += b[i] * prob[i][attack]
            for i in range(len(b)):
                b[i] = b[i] * prob[i][attack] / sum
            return b
        else :
            tmp = 0
            for i in range(len(belief)):
                e = (np.exp((self.ori_beliefa +ra[i]) / self.Belief_thr) - 1) / (math.e - 1)
                #e = (np.exp((self.ori_beliefa + ra[i]) / self.Belief_thr)) / (1-math.e )
                #e = math.log(self.ori_beliefa + ra[i]-ra_before[i] , self.Belief_thr)
                belief[i] =self.Min(1, e)

            sum = 0
            for i in range(len(belief2)):
                sum+= belief2[i]*prob[i][attack]
            for i in range(len(belief2)):
                belief2[i]=belief2[i]*prob[i][attack]/sum
            return belief,belief2



    def Strategy(self,beliefa,beliefd):
        rd11 = 0
        rd12 = 0
        rd21 = 0
        rd22 = 0
        ra11 = 0
        ra12 = 0
        ra21 = 0
        ra22 = 0
        for  i in range(len(self.Utility)+len(self.Fake_Utility)):
            ra11 += self.RewardM[0][0][1][i] * beliefa[i]
            ra12 += self.RewardM[1][0][1][i] * beliefa[i]
            ra21 += self.RewardM[0][1][1][i] * beliefa[i]
            ra22 += self.RewardM[1][1][1][i] * beliefa[i]

        beliefd_a1=self.Belief_Update([],None,beliefd,0,None)
        beliefd_a2 = self.Belief_Update([],None,beliefd,1,None)
        rd11 = self.RewardM[0][0][0] * beliefd_a1[0]  - self.CostM[0]
        rd12 = self.RewardM[1][0][0] * beliefd_a1[0] - self.CostM[1]
        rd21 = self.RewardM[0][1][0] * beliefd_a2[0] - self.CostM[0]
        rd22 = self.RewardM[1][1][0] * beliefd_a2[0] - self.CostM[1]

        sd = (ra22-ra12) / (ra11 -ra12-ra21+ra22)
        sa = (rd22-rd21) / (rd11- rd12-rd21+rd22)

        return [sa,1-sa],[sd,1-sd]

    def ChooseAct(self,strategy):
        rand=random.random()
        for i in range(len(strategy)):
            if rand > strategy[i]:
                rand -= strategy[i]
            else:
                    return i

  

    def Onepiece(self):
        sum_ra=0
        sum_rd =0
        sum_cost =0

        for i in range(self.Iteration):
            self.Belief_list = [[], []]
            e = (np.exp(self.ori_beliefa / self.Belief_thr) - 1) / (math.e - 1)
            #e = math.log(self.ori_beliefa, self.Belief_thr)
            beliefa = [e for i in range(len(self.Utility) + len(self.Fake_Utility))]
            beliefd = [self.ori_beliefd, 1 - self.ori_beliefd]
            self.Belief_list[0].append([e for i in range(len(self.Utility) + len(self.Fake_Utility))])
            self.Belief_list[1].append([self.ori_beliefd, 1 - self.ori_beliefd])
            ra = 0
            ba_before =[0 for i in range(len(self.Utility) + len(self.Fake_Utility))]
            ba = [0 for i in range(len(self.Utility) + len(self.Fake_Utility))]
            ra_sum = 0
            rd = 0
            cost = 0
            history = []
            item = 1

            while True:

                print(''.format(item))
                sa, sd = self.Strategy(beliefa, beliefd)
                
                
                print('  第{}阶段攻击者策略：{}，防御者策略：{}'.format(item, sa, sd))
                attack = self.ChooseAct(sa)

                defend = self.ChooseAct(sd)


                print('  第{}阶段先验信念{}{}'.format(item, beliefa, beliefd))
                print('  攻击者动作{}，防御者动作{}'.format(attack,defend))
                if cost + self.CostM[defend] <= self.Budget:
                    cost += self.CostM[defend]
                    history.append(attack + 3)
                    history.append(defend + 1)

                    for j in range(len(self.Utility) + len(self.Fake_Utility)):
                        ba[j] += self.BeliefM[defend][attack][1][j] * beliefa[j]
                        ra += self.RewardM[defend][attack][1][j] * beliefa[j]
                        ra_sum += self.BeliefM[defend][attack][1][j] * beliefa[j]

                    beliefa, beliefd = self.Belief_Update(beliefa, ba, beliefd, attack, ba_before)
                    self.Belief_list[0].append(copy.deepcopy(beliefa))
                    self.Belief_list[1].append(copy.deepcopy(beliefd))
                    ba_before = copy.deepcopy(ba)
                    print('  第{}阶段后验信念{}{}'.format(item, beliefa, beliefd))
                    rd += self.RewardM[defend][attack][0] * beliefd[0] - self.CostM[defend]
                    if ra_sum >= self.AttackerTarget:
                        break
                    item += 1
         
                else:
                    history.append(4)
                    history.append(0)
                    for j in range(len(self.Utility)):
                        ba[j] += self.Sysnum[j]*self.Utility[j]*self.Attack_para[1]* beliefa[j]
                        ra += self.Sysnum[j]*self.Utility[j]*self.Attack_para[1]* beliefa[j]
                        ra_sum += self.Sysnum[j]*self.Utility[j]*self.Attack_para[1]* beliefa[j]
                        rd -=self.Sysnum[j]*self.Utility[j]*self.Attack_para[1]*beliefd[0]
                    beliefa, beliefd = self.Belief_Update(beliefa, ba, beliefd, attack, ba_before)
                    self.Belief_list[0].append(copy.deepcopy(beliefa))
                    self.Belief_list[1].append(copy.deepcopy(beliefd))
                    ba_before = copy.deepcopy(ba)
                    print('  第{}阶段后验信念{}{}'.format(item, beliefa, beliefd))
                    #rd += self.RewardM[defend][attack][0] * beliefd[0] - self.CostM[defend]
                    if ra_sum >= self.AttackerTarget:
                        break
                    item += 1

            sum_ra+=ra
            sum_rd+=rd
            sum_cost+=cost

            if item>1 :
                s = ''
                s_ = ['', '', '', '', '', '', '', '', '', '']
                for k in range(len(self.Belief_list[1])):
                    s += str(self.Belief_list[1][k][0]) + ','
                print(s)
                for k in range(len(self.Belief_list[0])):
                    for j in range(len(self.Utility) + len(self.Fake_Utility)):
                        s_[j] += str(self.Belief_list[0][k][j]) + ','
                for k in range(len(self.Utility) + len(self.Fake_Utility)):
                    print(s_[k])
            print('第{}次迭代动作序列为{}，攻击者收益{}，防御者收益{}，防御成本{}'.format(i+1,history, ra, rd, cost))



        print('平均攻击收益：{}，平均防御收益：{}，平均防御成本：{}'.format(sum_ra/self.Iteration,sum_rd/self.Iteration,sum_cost/self.Iteration))
        print(self.Utility)
        print(self.Fake_Utility)

        




if __name__ == '__main__':
    G=Game()
    G.Onepiece()

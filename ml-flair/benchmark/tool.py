import numpy as np
import random
import math

random.seed(666)
def OCF(ci, time, power, n_user):
    # ci : g/kWh
    # time : seconds
    # power : W
    # return : CO2(mg)
    return power / 1000 * time / 3600 * ci / n_user * 1000

def ECF(time, ecf, lifetime, n_user):
    # time : second
    # lifetime : second
    # ecf : kg
    # return : CO2(mg)
    return ecf * 1000 * 1000 * time / lifetime / n_user


def Energy(time, power, n_user=1):
    return power / 1000 * time / 3600 / n_user


def choose_client_id(lst, num_all_user, client_per_round):
    window_size = client_per_round * 2 # window_size = 2 * client_per_round
    index = random.randint(0, num_all_user - 1)
    client_lst = range(index, index + window_size)
    if index + window_size > num_all_user:
        client_lst = (list(range(index, num_all_user)) + list(range(0,index + window_size - num_all_user)))
    client_lst = random.sample(client_lst, client_per_round)

    result = []
    for client in client_lst:
        result.append(lst[client])
    return result


def choose_client_id_greedy(lst, cur_round_num):
    clients_per_round = 50
    num_all_user = len(lst)

    index = cur_round_num % (3050 // clients_per_round) * clients_per_round
    # print(cur_round_num, index, index + clients_per_round)
    if index + clients_per_round > num_all_user:
        return lst[index:]
    return lst[index:index + clients_per_round]


def max_time(user_data: dict, chosen_user_list, model: str):

    high_data_max, mid_data_max, low_data_max = -np.inf, -np.inf, -np.inf

    for user in chosen_user_list:
        data_num = min(512, user_data[user]['data_num'])  # 최대 512개까지 허용
        device = user_data[user]['device']
        #device = 'high'

        if device == 'high' and high_data_max < data_num:
            high_data_max = data_num
        elif device == 'mid' and mid_data_max < data_num:
            mid_data_max = data_num
        elif device == 'low' and low_data_max < data_num:
            low_data_max = data_num

    high_max_exetime = data.time_comp(device="high", model=model, epochs=2, data_num=high_data_max)
    mid_max_exetime = data.time_comp(device="mid", model=model, epochs=2, data_num=mid_data_max)
    low_max_exetime = data.time_comp(device="low", model=model, epochs=2, data_num=low_data_max)

    return max(high_max_exetime, mid_max_exetime, low_max_exetime)


def make_user_cooltime(user_data: dict):
    # user_data: client 정보들
    # alpha: cooltime 조절

    user_data = dict(sorted(user_data.items(), key=lambda x: x[1]['data_num'], reverse=True))
    user_credits = dict()

    for user_id in user_data:
        user_credits[user_id] = 0

    return user_credits

def make_user_credits(user_data: dict):
    # user_data: client 정보들
    # alpha: credit 조절

    max_carbon_intensity = 878.5315025

    user_data = dict(sorted(user_data.items(), key=lambda x: x[1]['data_num'], reverse=True))
    user_credits = dict()

    # for user_id in user_data:
    #     ci = user_data[user_id]['CI']
    #     user_credits[user_id] = 1/ci * max_carbon_intensity

    for user_id in user_data:
        ci = user_data[user_id]['CI']
        user_credits[user_id] = 0

    return user_credits


def add_user_credits(user_data: dict, user_credits: dict, alpha: float):
    # user_data: client 정보들
    # alpha: credit 조절

    for user_id in user_data:
        ci = user_data[user_id]['CI']
        user_credits[user_id] += 1 / ci * alpha

    return user_credits


def delete_credit_zero(user_credits: dict):

    delete_id_list = []

    for user_id in user_credits:
        if user_credits[user_id] == 0:
            delete_id_list.append(user_id)

    for user_id in delete_id_list:
        del(user_credits[user_id])

    return user_credits


def cooltime_minus_one(user_cooltime: dict):

    for user_id in user_cooltime:
        if user_cooltime[user_id] > 0:
            user_cooltime[user_id] -= 10
            user_cooltime[user_id] = max(0, user_cooltime[user_id])


def make_IQR(user_data: dict, user_cooltime: dict):

    user_ids = []
    for user_id in user_cooltime:
        if user_cooltime[user_id] == 0:
            user_ids.append(user_id)

    N = len(user_ids)

    data_num_acc = [0 for i in range(513)] # data개수 분포
    data_num_lst = [] # user당 data 개수

    for user_id in user_ids:
        data_num = min(512, user_data[user_id]['data_num'])
        data_num_acc[data_num] += 1
        data_num_lst.append(data_num)

    data_num_lst.sort()

    Q1 = data_num_lst[int(N / 4)]
    Q2 = data_num_lst[int(N / 4 * 2)]
    Q3 = data_num_lst[int(N / 4 * 3)]

    IQR = Q3 - Q1
    delta = 1.5

    LO = Q1 - delta * IQR
    UO = Q3 + delta * IQR

    outlier = math.floor(UO) + 1

    data_sum = 0
    client_sum = 0 # non-upper-class

    # non-outlier count
    for i in range(1, outlier):
        data_sum += i * data_num_acc[i]
        client_sum += data_num_acc[i]

    return user_ids, N - client_sum


def choose_client_id_with_credits(user_data: dict, user_credits: dict, client_cluster: list, K: int, alpha):

    N = len(client_cluster)
    final_client_set = set()

    # count credits
    num_over_one_credits = 0
    for user_id in client_cluster:
        if user_credits[user_id] >= 1:
            num_over_one_credits += 1

     # 해당 cluster의 credits이 부족하다면
    while num_over_one_credits < K:
        for user_id in user_data:
            ci = user_data[user_id]['CI']
            user_credits[user_id] += 1 / ci

        num_over_one_credits = 0
        for user_id in client_cluster:
            if user_credits[user_id] >= 1:
                num_over_one_credits += 1

    # choose real client
    while len(final_client_set) < K:
        index = random.randint(0, N-1)
        user_id = client_cluster[index]
        if user_credits[user_id] >= 1:
            # print(user_credits[user_id])
            final_client_set.add(user_id)

    # minus credits
    for user_id in final_client_set:
        user_credits[user_id] -= 1

    return [user_id for user_id in final_client_set]


def choose_client_id_with_cooltime(user_data: dict, user_cooltime: dict, client_cluster: list, K: int, alpha):

    chosen_user_list = set()

    while len(chosen_user_list) < K:

        [cur_client_id] = random.sample(client_cluster, 1)

        if user_cooltime[cur_client_id] > 0:
            user_cooltime[cur_client_id] -= 1
        else:
            chosen_user_list.add(cur_client_id)
            user_cooltime[cur_client_id] += int(user_data[cur_client_id]["CI"] / alpha)

    return chosen_user_list

def data_group_ok():
    a = random.random() * 100
    if a <= 3:
        return True
    else:
        return False
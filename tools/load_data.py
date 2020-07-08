import os
import numpy as np
import collections


class LoadData:
    def __init__(self, args):
        self.args = args
        self.data_path = os.path.join(args.base_path, 'data/')

    def load_data(self):
        # user_history_dict has all users and their corresponding positive rated items 
        train_data, test_data, user_history_dict = self.load_rating()

        n_entity, n_relation, kg = self.load_kg()
        ripple_set = self.get_ripple_set(kg, user_history_dict)
        return train_data, test_data, n_entity, n_relation, ripple_set

    def load_rating(self):
        print('reading rating file ...')

        # reading rating file into a numpy array
        # e.g. rating_np.shape for movie dataset is (753774, 3)
        rating_file = self.data_path + self.args.dataset + '/ratings_final'
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)
        
        print('splitting dataset ...')

        # train:test = 6:2
        test_ratio = 0.2
        n_ratings = rating_np.shape[0]  # total number of ratings, movie 753774

        # get the test ratings indices 20%
        test_indices = np.random.choice(n_ratings,
                                        size=int(n_ratings * test_ratio),
                                        replace=False)
        train_indices = set(range(n_ratings)) - set(test_indices)  # train 80%

        # traverse training data, only keeping the users with positive ratings
        user_history_dict = dict()
        for i in train_indices:
            user = rating_np[i][0]  # user
            item = rating_np[i][1]  # item
            rating = rating_np[i][2]  # rating 1 or 0
            if rating == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = []
                user_history_dict[user].append(item)
        
        # user_history_dict has all users and their corresponding positive rated items 
        train_indices = [i for i in train_indices
                         if rating_np[i][0] in user_history_dict]
        test_indices = [i for i in test_indices
                        if rating_np[i][0] in user_history_dict]

        train_data = rating_np[train_indices]
        test_data = rating_np[test_indices]

        return train_data, test_data, user_history_dict

    def load_kg(self):
        print('reading KG file ...')

        # reading kg file
        kg_file = self.data_path + self.args.dataset + '/kg_final'
        if os.path.exists(kg_file + '.npy'):
            kg_np = np.load(kg_file + '.npy')
        else:
            kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg_np)

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))  # unique number of entities
        n_relation = len(set(kg_np[:, 1]))  # unique number of relations

        print('constructing knowledge graph ...')
        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((tail, relation))

        return n_entity, n_relation, kg

    def get_ripple_set(self, kg, user_history_dict):
        print('constructing ripple set ...')

        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails),
        #          (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        ripple_set = collections.defaultdict(list)

        for user in user_history_dict:
            for h in range(self.args.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                """
                if the current ripple set of the given user is empty,
                we simply copy the ripple set of the last hop here
                this won't happen for h = 0,
                because only the items that appear in the KG have been selected
                this only happens on 154 users in Book-Crossing dataset
                (since both book dataset and the KG are sparse)
                """
                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.args.n_memory
                    indices = np.random.choice(
                        len(memories_h),
                        size=self.args.n_memory,
                        replace=replace
                        )
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append(
                        (memories_h, memories_r, memories_t)
                        )

        return ripple_set

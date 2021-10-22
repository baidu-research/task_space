"""
Create .npz file that stores the data indices seen by the admin and agents
"""

import argparse
import numpy as np
import torchvision



def take_set(train_or_test_set, class_ids):
    """
        Return data ids for a list of given class ids

        Input:
        train_or_test_set: train or test partition of cifar100, in specific:
            torchvision.datasets.CIFAR100('images/', train=True)
            or
            torchvision.datasets.CIFAR100('images/', train=False)

        class_ids: a list of class ids

        Return:
        indices of samples whose class labels are in `class_ids`
    """
    train_or_test_id = []
    for c in class_ids:
        ids = np.where(np.asarray(train_or_test_set.targets) == c)[0]
        train_or_test_id.append(ids)
    return np.hstack(train_or_test_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--admin-classes', type=int, default=30,
                        help='Number of classes that each admin task has to '
                        'classifiy (default: 30)')
    parser.add_argument('--admin-tasks', type=int, default=1,
                        help='number of admin tasks (default: 1). If >1, '
                        'The tasks are guaranteed to be different from each other')
    parser.add_argument('--admin-samples', type=int, default=10,
                        help='number of per-class training samples seen '
                        'by admin. Supposed to be small. (default: 10)')
    parser.add_argument('--probe-samples', type=int, default=10,
                        help='number of samples reserved (per class) as '
                        'probing samples (default: 10). The labels of these '
                        'samples are not used.')
    
    parser.add_argument('--agents', type=int, default=100,
                        help='number of agents, default to 100')
    parser.add_argument('--agent-classes', type=int, default=30,
                        help='classes seen by each agent')

    parser.add_argument('--nonoverlap-classes-for-agents', type=int, default=0,
                        help='random split out these many classes for agents, '
                        'then each agent takes `agent-classes` as its task. '
                        'Default to 0, i.e., the union of agents '
                        'see all 100 classes')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save', type=str, default='split.npz',
                        help='name of a .npz to save the data indices')
    args = parser.parse_args()
    
    
    full_train = torchvision.datasets.CIFAR100('images/', train=True)
    full_test = torchvision.datasets.CIFAR100('images/', train=False)
    np.random.seed(args.seed)

    # From the training partition of all classes involved in admin's tasks, we reserve
    # 1) `probe-samples` per class as probing sample, here the classes are all
    #    classes involved in admin's tasks. So if
    #    nonoverlap-classes-for-agents = 0, probe-samples * 100 samples are used as probing data
    #    nonoverlap-classes-for-agents = 60, probe-samples * (100-60) samples are used as probing data
    # 2) `admin-samples` per class as training data for admin's tasks.
    #   The classes are, again, all classes involved in admin's tasks.
    probe_ids = [] # e.g. (100, 10) if overlap allowed, (40, 10) if no overlap
    reserve_admin_train_ids = [] # (100, 10) if overlap allowed, (40, 10) if no overlap
    if not args.nonoverlap_classes_for_agents:
        # indices of remaining training data (100, 500-20) if overlap allowed
        remaining_train_ids = []
    
    if args.nonoverlap_classes_for_agents:
        all_agent_classes = np.sort(np.random.choice(100,
            args.nonoverlap_classes_for_agents, replace=False))
        all_admin_classes = np.sort(np.fromiter(
            set(range(100)) - set(all_agent_classes), np.int64))
    else:
        all_agent_classes = range(100)
        all_admin_classes = range(100)

    # sample probing and training data for admin, the rest stored for agents
    for cls in all_admin_classes:
        ids = np.where(np.asarray(full_train.targets)==cls)[0]
        ids = np.random.permutation(ids)

        probe_ids.append(ids[:args.probe_samples])

        reserve_admin_train_ids.append(
            ids[args.probe_samples : args.probe_samples + args.admin_samples])

        if not args.nonoverlap_classes_for_agents: # no overlap
            remaining_train_ids.append(
                ids[args.probe_samples + args.admin_samples:])

    probe_ids = np.hstack(probe_ids)
    reserve_admin_train_ids = np.vstack(reserve_admin_train_ids)
    if not args.nonoverlap_classes_for_agents:  # no overlap
        remaining_train_ids = np.vstack(remaining_train_ids)
    
    # create T tasks, each an N-way classification
    admin_classes = np.zeros((args.admin_tasks, args.admin_classes), dtype=int)
    admin_train_ids = np.zeros((args.admin_tasks,
                                args.admin_classes*args.admin_samples),
                               dtype=int)
    admin_test_ids = np.zeros((args.admin_tasks,
                               args.admin_classes*100), dtype=int)
    for task_id in range(args.admin_tasks):
        this_classes = np.random.permutation(all_admin_classes)[:args.admin_classes]
        this_classes.sort()
        admin_classes[task_id] = this_classes
        this_classes_row_id = [_ in this_classes for _ in all_admin_classes]
        admin_train_ids[task_id] = reserve_admin_train_ids[this_classes_row_id].ravel()
        admin_test_ids[task_id] = take_set(full_test, this_classes) 

    if args.admin_tasks > 1: # ensure no two admin tasks are the same
        for i in range(args.admin_tasks):
            for j in range(i+1, args.admin_tasks):
                assert set(admin_classes[i].tolist()) != \
                       set(admin_classes[j].tolist())

    # Agents take their training and test data
    agent_classes = [] # classes seen by each agents, (agents, classes_per_agent)
    for a in range(args.agents):
        agent_classes.append(
            np.random.choice(all_agent_classes, args.agent_classes, False))
    
    # ensure no two agents see completely same set of classes
    for i in range(args.agents):
        for j in range(i+1, args.agents):
            assert set(agent_classes[i].tolist()) != set(agent_classes[j].tolist())
    agent_classes = np.vstack(agent_classes)
    agent_train_ids = []
    agent_test_ids = []
    
    # ensure none of agents' tasks is same with any of admin's tasks
    if args.nonoverlap_classes_for_agents == 0:
        for i in range(args.agents):
            for j in range(args.admin_tasks):
                assert set(agent_classes[i].tolist()) != set(admin_classes[j].tolist())

    for a in range(args.agents):
        if args.nonoverlap_classes_for_agents == 0:
            agent_train_ids.append(remaining_train_ids[agent_classes[a]].ravel())
        else:
            agent_train_ids.append(take_set(full_train, agent_classes[a]))
        agent_test_ids.append(take_set(full_test, agent_classes[a]))

    agent_train_ids = np.vstack(agent_train_ids)
    agent_test_ids = np.vstack(agent_test_ids)

    np.savez(args.save,
             admin_probe_ids=probe_ids,
             admin_train_ids=admin_train_ids,
             admin_test_ids=admin_test_ids,
             admin_classes=admin_classes,
             agent_train_ids=agent_train_ids,
             agent_test_ids=agent_test_ids,
             agent_classes=agent_classes)

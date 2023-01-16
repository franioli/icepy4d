import multiprocessing
from multiprocessing import Pool


def worker(procnum):
    '''worker function'''
    print(f'{procnum} represent!')
    return procnum


if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(jobs)
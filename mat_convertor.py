import scipy.io as sio
import os

if __name__ == "__main__":
    # mat = sio.loadmat("/Users/Ganymedian/Desktop/dynamic-rw/datasets/academic_confs.mat")

    mat = sio.mmread("/Users/Ganymedian/Desktop/ca-coauthors-dblp/ca-coauthors-dblp.mtx")
    print(mat)
    # graph = mat['network'].toarray()
    #
    # srcs = graph.shape[0]
    # dests = graph.shape[1]
    #
    # with open(os.path.join("/Users/Ganymedian/Desktop/dynamic-rw/datasets/", "cocit-edge-list.txt"),
    #           "a") as f:
    #     for x in range(0, srcs):
    #         for y in range(x, dests):
    #             e = graph[x, y]
    #             if e == 1:
    #                 f.write("{}\t{}\n".format(x, y))



                    # nodes = labels.shape[0]
                    # types = labels.shape[1]
                    # v_labels = {}
                    # for x in range(0, nodes):
                    #     v_labels[x] = -1
                    #
                    # s = 0
                    # for x in range(0, nodes):
                    #     for y in range(0, types):
                    #         v_t = labels[x, y]
                    #         if v_t == 1:
                    #             v_labels[x] = y
                    #
                    # with open(os.path.join("/Users/Ganymedian/Desktop/dynamic-rw/datasets/", "cocit-labels.txt"),
                    #           "a") as f:
                    #     for key, value in v_labels.items():
                    #         f.write("{}\t{}\n".format(key, value))


                    # print(sio.whosmat("/Users/Ganymedian/Desktop/dynamic-rw/datasets/academic_confs.mat"))
                    # print()
                    # graph = mat['network']
                    # print(graph.toarray().shape)

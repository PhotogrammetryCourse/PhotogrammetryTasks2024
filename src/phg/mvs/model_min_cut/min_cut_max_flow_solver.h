#pragma once

#include "bk_solver/graph.h"

template <typename node_t>
class MaxFlow
{
    typedef float value_t;

public:
    typedef Graph<value_t, value_t, value_t> graph_t;

public:
    MaxFlow(size_t numNodes) : graph((int) numNodes, 4 * (int) numNodes), numNodes(numNodes)
    {
    }

    inline void addNode(node_t index, value_t source, value_t sink)
    {
        rassert(source >= 0 && sink >= 0, 2734848121234021);

        node_t node_id = graph.add_node();
        rassert(node_id == index, 37281310023);
        graph.add_tweights(node_id, source, sink);
    }

    inline void addEdge(node_t n1, node_t n2, value_t capacity, value_t reverseCapacity)
    {
        rassert(n1 >= 0 && n1 < numNodes, 2382913030);
        rassert(n2 >= 0 && n2 < numNodes, 2382913031);
        rassert(n1 != n2, 2382913032);
        rassert(capacity >= 0 && reverseCapacity >= 0, 75871241026);
        graph.add_edge(n1, n2, capacity, reverseCapacity);
    }

    value_t computeMaxFlow()
    {
        return graph.maxflow();
    }

    inline bool isNodeOnSrcSide(node_t n)
    {
        return (graph.what_segment(n) == graph_t::SOURCE);
    }

protected:
    graph_t graph;
    size_t numNodes;
};
#ifndef PARTITION_H
#define PARTITION_H

#define _GNU_SOURCE

#include "struct.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define K 5

typedef unsigned long (*partition)(adjlist*,unsigned long*);

unsigned long init(adjlist *g,unsigned long *lab);

unsigned long louvain(adjlist *g, unsigned long *lab);

unsigned long louvainComplete(adjlist *g, unsigned long *lab);


/**
   Returns the weighted degree of a node, i.e., the sum of all weights of edges connected to it
   @param graph The graph to be displayed
   @param node The node whose degree must be calculated
   @return the weighted degree of the node
*/

#define MIN_IMPROVEMENT 0.005

// louvain partition
typedef struct {
  // size of the partition
  unsigned long size;
  
  // community to which each node belongs
  unsigned long *node2Community;
  
  // in and tot values of each node to compute modularity 
  long double *in;
  long double *tot;
  
  // utility arrays to find communities adjacent to a node
  // communities are stored using three variables
  // - neighCommWeights: stores weights to communities
  // - neighCommPos: stores list of neighbor communities
  // - neighCommNb: stores the number of neighbor communities
  long double *neighCommWeights;
  unsigned long *neighCommPos;
  unsigned long neighCommNb;
} louvainPartition;


unsigned long louvain(adjlist *g,unsigned long *lab);

//inline long double degreeWeighted(adjlist *g, unsigned long node);


/**
    Returns the weight of the self-loop of a node
    Assumes that there is at most one self node for a given node
    @param graph The graph to be displayed
    @param node The node whose self-loop weight must be calculated
    @return the self-loop weight
*/
//inline long double selfloopWeighted(adjlist *g, unsigned long node);

/**
    Frees a louvain partition and all pointers attached to it
    @param p The Louvain partition
    @return nothing
*/
void freeLouvainPartition(louvainPartition *p);

/**
    Creates a louvain partition from a graph and initializes it
    @param g The graph for which a partition has to be createed
    @return The louvain partition
*/
louvainPartition *createLouvainPartition(adjlist *g);

/**
    Computes modularity of the given graphPartitio
    @param p The Louvain partition
    @param g The partitionned graph
    @return The modularity of the partition
*/
long double modularity(louvainPartition *p, adjlist *g);

/**
   Computes the set of neighboring communities of node
   @param p The Louvain partition
   @param g the partitionned graph
   @param node The node whose neighbor communities must be computed
   @return Nothing
*/
void neighCommunities(louvainPartition *p, adjlist *g, unsigned long node);

// updates a given partition with the current Louvain partition
unsigned long updateGraphPartition(partition *p, unsigned long *part, unsigned long size);

// generates the binary graph of communities as computed by one_level
adjlist* louvainPartition2Graph(louvainPartition *p, adjlist *g);

/**
    Computes one level of Louvain (iterations over all nodes until no improvement
    @param p The Louvain partition
    @param g The partitionned graph
    @param minImprovement The minimal improvement under which the process stops
    @return the increase of modularity during the level
*/
long double louvainOneLevel(louvainPartition *p, adjlist *g);

/**
    Computes a partition with the first level of Louvain method
    @param g The graph to be partitionned
    @param part The final partition
    @return nothing
*/
unsigned long louvain(adjlist *g, unsigned long *part);

/**
    Computes a partition with the Louvain method
    @param g The graph to be partitionned
    @param part The final partition
    @return nothing
*/
unsigned long louvainComplete(adjlist *g, unsigned long *part);


/**
    Removes a node from its community and update modularity
    @param p The Louvain partition
    @param g the partitionned graph
    @param node The node to be removed
    @param comm The community to which node belongs
    @param dnodecomm The weighted degree from node to comm
    @return nothing
*/
//inline void removeNode(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm, long double dnodecomm);

/**
    Adds a node to a community and update modularity
    @param p The Louvain partition
    @param g the partitionned graph
    @param node The node to be added
    @param comm The community to which node must be added
    @param dnodecomm The weighted degree from node to comm
    @return nothing
*/
//inline void insertNode(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm, long double dnodecomm);

/**
    Computes the increase of modularity if a node where to be added to a given community
    - Note that node itself is not usefull
    - Note that the increase is not the real increase but ensures that orders are respected
    @param p The Louvain partition
    @param g the partitionned graph
    @param comm The community that is modified
    @param dnodecomm The weighted degree from node to comm
    @param nodeDegree The weighted degree of the node
    @return nothing
*/
//inline long double gain(louvainPartition *p, adjlist *g, unsigned long comm, long double dnodecomm, long double nodeDegree);



#endif

# 210. Course Sechdule II

"""

Return the ordering of courses you should take to finish all courses, in case of many answers , any will do. If it is not possible to finish all courses return an empty array. 

Here is the structure of the input:

You have to take a number of courses , 0-index labelling, and your are given an array of prerequisities that contains pairs [a,b] indicating a must be taken before b. 


"""

def course_schedule_feasibility_optim_strict(dependencies, n):

    """

    This is straightforward topological sorting

    """


    graph = {i: [] for i in range(n)}          # ensure every vertex exists
    indegree = {i: 0 for i in range(n)}        # start with 0 incoming edges

    # Construct the graph adjacency list
    for j,i in dependencies:
        graph[j].append(i)
        indegree[i] += 1 


    # Go over the graph and find any nodes without incoming or outgoing, they can be done first ( or last too )


    result = []
    tovisit = []

    for k,v in indegree:
        if v == 0:
            tovisit.append(k)

    while tovisit:

        cursor = tovisit.pop()

        result.append(cursor)

        for i in graph[cursor]:
            indegree[i] -= 1
            if indegree[i] == 0 :
                tovisit.append(i)

    if len(result) != n:
        return []

    return tovisit



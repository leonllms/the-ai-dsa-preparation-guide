# 13. **Trie (Prefix Tree)**

**Problem:**  
Given a dictionary of *d* words, build a trie and then write a routine that, for any query string *q*, returns **all words** in the dictionary that have *q* as a prefix.


**Idea**

A trie (prefix tree) stores each word character by character.  
All words that share a prefix follow the same path from the root.  
To answer a query we walk the prefix, then output every word that can be
reached from the node where the walk ends.

**Building the trie**

1. Start with an empty root node.  
2. For each word in the dictionary  
   * begin at the root.  
   * for each character `c` in the word  
        – if there is already a child node for `c`, move to it;  
        – otherwise create a new child node for `c` and move to it.  
   * after the last character mark the current node as “end‑of‑word”.  
   * (Optionally store the whole word at that node for easy retrieval.)

**Answering a query `q`**

1. Start at the root and follow the characters of `q`.  
   * If at any step a needed child does not exist, no dictionary word has
     that prefix → return an empty list.  
2. When the whole prefix has been consumed, you are at the node that
   represents `q`.  
3. Perform a depth‑first (or breadth‑first) traversal from this node:  
   * Whenever a visited node is marked “end‑of‑word”, output the word
     formed by the path from the root to that node (or the stored word).  
   * Continue exploring all children recursively.  
4. Collect all emitted words; they are exactly the dictionary words that
   start with `q`.

**Complexities**

*Building*: O(total number of characters in all words).  
*Query*: O(length of `q` + number of output characters).  
The traversal after the prefix costs only for the matching subtree,
not for the whole dictionary.

**Summary**

Insert every dictionary word into a character‑by‑character tree, marking
where words end. To find all words with a given prefix, walk the prefix,
then list every complete word reachable from that node by a simple tree
search. This yields fast prefix look‑ups and avoids scanning the whole
dictionary.


## Implementation

Provided a corpus of words we map each word to represent a walk on the tree. All
words that share a common prefix need to follow the same path from the root up
to the length of the common prefix. 

There are two main methods of interest, search, and insert. We will extend to 
delete afterwards. Search walks the tree for the query string and returns all 
paths of the subtree, i.e. all completions.

The insertions follow a similar logic, walk up to the node matching the word
in the tree ( i.e. find the prefix ) and then insert the remaining part.

```python
from typing import Dict, List, Tuple

class prefix_node:

    def __init__(self, value, children: Dict[ str, prefix_node] | None = None, prefix: str | None = None):     
        self._children = children if children is not None else {}
        self._value = value
        self._prefix = prefix

    def value(self):
        return self._value

    def find_child(self, value):
        return self._children.get(value, None)

    def all_children(self) -> List['prefix_node']:
        return list(self._children.values())

    def full_prefix(self) -> str:
        return self._prefix

    def store_prefix(self, key: str) -> str:
        self._prefix = key

    def add_child(self, c) -> 'prefix_node':
        ref = self._children.get(c, None)
        if ref is not None:
            return ref
        node = prefix_node(c)
        self._children[c] = node
        return node


class prefix_tree:

    def __init__(self, root: prefix_node | None = None):
        self.root = root if root is not None else prefix_node(None)

    def _advance_to_prefix_end(self, query: str) -> Tuple[int | None, prefix_node | None]:
        """

        Advances up to the last node of the common prefix between the query
        and the trie

        """

        if not self.root.all_children() or len(query) == 0:
            return None, None

        node = self.root.find_child(query[0]):
        if node is None:
            return None, None

        i = 0

        def advance():
            """
            Looks at the next character in the input string and advances node to
            the child that has an equal value, returning true, otherwise False.

            """
            nonlocal node
            ref = node.find_child(query[i+1])
            if ref is not None:
                node = ref 
                return True
            return False

        # In this loop we advance until there is no more characters to check or
        # there are no more equal characters in the tree, or we have reached
        # the end of the tree, i.e. all characters match.
        while i < len(query)-1 and advance():
            i+=1

        return i, node


    def search(self,query: str, limit: int = 10) -> List[str]:
        # start from the root walking down until have exchausted the query. Then
        # perform DFS to retrieve up to limit number of suffixes and return them.

        begin, node = self._advance_to_prefix_end(query)
        if begin is None:
            return []
        begin += 1

        if begin != len(query):
            return []

        completions = []
        tovisit = [n for n in node.all_children()]
        while limit>0 and tovisit:
            limit-=1 
            node = tovisit.pop()
            children = node.all_children()

            if children:
                tovisit.extend(children)
            else:
                completion = node.full_prefix()
                completions.append(completion[begin:])

        return completions


    def insert(self, key: str) -> bool:

        if len(key) == 0:
            return False

        # Four cases for begin:
        # It is None because: 
        # 1. no starting match, 
        # 2. query is empty string
        # 3. tree is empty
        # Not None so we need to insert the remaing
        begin , node = self._advance_to_prefix_end(key)
        if begin is None:
            node = self.root.add_child(key[0])
            begin = 0

        i = begin + 1
        while i < len(key):
            node = node.add_child(key[i])
            i+=1
        node.store_prefix(key)
        return True
                


def build_prefix_tree(corpus : List[ str ]):
    newtree = prefix_tree()

    for sentence in corpus:
        for word in sentence.split():
            newtree.insert(word)

    return newtree

```

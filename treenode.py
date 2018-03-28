def _comparison_tuple(node):
    return node.data['upostag'], \
           node.data['xpostag'], \
           node.data['deprel'], \
           node.data['lemma']

def eq(nodea, nodeb):
    return _comparison_tuple(nodea) == _comparison_tuple(nodeb)

def lt(nodea, nodeb):
    return _comparison_tuple(nodea) < _comparison_tuple(nodeb)

def get_descendants(node, include_self=False):
    # Non-recursive tree traversal algorithm. Taken from https://softwareengineering.stackexchange.com/a/226162/161912
    result = []
    stack = [node]
    while stack:
        desc = stack.pop()
        if desc != node or include_self:
            result.append(desc)
        for child in desc.children:
            stack.append(child)
    return result

def get_ordered_descendants(node, include_self=False):
    assert not include_self

    result = get_descendants(node, include_self)
    result.sort(key=_comparison_tuple)
    return result

def matching_descendants(nodea, nodeb, include_self=False):
    assert not include_self

    pairs = []

    i, j = 0, 0
    nodesa = get_ordered_descendants(nodea, include_self)
    nodesb = get_ordered_descendants(nodeb, include_self)
    na, nb = len(nodesa), len(nodesb)
    
    while i < na and j < nb:
        if lt(nodesb[j], nodesa[i]):
            j += 1
        elif lt(nodesa[i], nodesb[j]):
            i += 1
        else: # The nodes match
            assert eq(nodesa[i], nodesb[j])
            j_old = j
            while True:
                while True:
                    pair = nodesa[i], nodesb[j]
                    pairs.append(pair)
                    j += 1

                    if not (j < nb and eq(nodesa[i], nodesb[j])):
                        break

                i += 1
                j_final = j
                j = j_old

                if not (i < na and eq(nodesa[i], nodesb[j])):
                    break

            j = j_final
    
    return pairs

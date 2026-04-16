def separate4(widths, l=0.1, r=0.9):
    def get_state(x):
        return 'open' if x > r else ('close' if x < l else 'mid')
    states = [get_state(x) for x in widths]
    i = 0
    while i < len(states):
        if states[i] == 'mid':
            pre = states[i-1] if i > 0 else 'open'
            j = i
            while j < len(widths) and states[j] == 'mid':
                j += 1
            nxt = states[j] if j < len(widths) else 'open'
            if pre == nxt:
                states[i:j] = [pre] *(j-i)
            elif pre == 'open':
                states[i:j] = ['closing'] *(j-i)
            else:
                states[i:j] = ['openning'] *(j-i)
            i = j
        else:
            i += 1
    phase_start = []
    for i in range(len(states)):
        if i == 0 or states[i] != states[i-1]:
            phase_start.append(i)
    return phase_start, states

def separate(widths, mid=0.5):
    def get_state(x):
        return 'open' if x > mid else 'close'
    states = [get_state(x) for x in widths]

    phase_start = []
    for i in range(len(states)):
        if i == 0 or states[i] != states[i-1]:
            phase_start.append(i)
    return phase_start, states


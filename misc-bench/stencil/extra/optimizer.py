# Finds the optimal layout for stencil assuming intra memory-layer transfer cost is consistent
#   ie., all subarray to subarray transfers are equivalent in cost (within a bank)
# Note after running for some test values: layout does change based on transfer cost parameters

subarray_block_width = 100
subarray_block_height = 100
subarrays_per_bank = 16
banks_per_rank = 16
ranks = 16
transfer_cost_subarray_to_subarray = 1
transfer_cost_bank_to_bank = 20
transfer_cost_rank_to_rank = 100


def get_stats(num_blocks, grid_width, block_width, block_height):

    if num_blocks % grid_width != 0:
        raise ValueError("num blocks must be divisible by grid width")

    grid_height = num_blocks/grid_width
    to_move_horizontal = (2 * grid_height * (grid_width - 1)) * (block_width - 2)
    to_move_vertical = (2 * grid_width * (grid_height - 1)) * (block_height - 2)
    to_move_diagonal = (4 * (grid_width-1) * (grid_height - 1))
    to_move_total = to_move_horizontal + to_move_vertical + to_move_diagonal
    width_next = grid_width * block_width
    height_next = grid_height * block_height
    return to_move_total, width_next, height_next

def total_move_cost(subarray_grid_width, bank_grid_width, rank_grid_width):
    to_move_s2s, bank_block_width, bank_block_height = get_stats(subarrays_per_bank, subarray_grid_width, subarray_block_width, subarray_block_height)
    to_move_b2b, rank_block_width, rank_block_height = get_stats(banks_per_rank, bank_grid_width, bank_block_width, bank_block_height)
    to_move_r2r, final_block_width, final_block_height = get_stats(ranks, rank_grid_width, rank_block_width, rank_block_height)
    cost = transfer_cost_subarray_to_subarray*to_move_s2s
    cost += transfer_cost_bank_to_bank*to_move_b2b
    cost += transfer_cost_rank_to_rank*to_move_r2r
    return cost

def get_divisors(n):
    """Get all divisors of n"""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


# Find all valid divisors
subarray_divisors = get_divisors(subarrays_per_bank)
bank_divisors = get_divisors(banks_per_rank)
rank_divisors = get_divisors(ranks)

print("Valid divisors:")
print(f"  subarrays_per_bank ({subarrays_per_bank}): {subarray_divisors}")
print(f"  banks_per_rank ({banks_per_rank}): {bank_divisors}")
print(f"  ranks ({ranks}): {rank_divisors}")
print()

# Find optimal configuration
min_cost = float('inf')
best_config = None

for sgw in subarray_divisors:
    for bgw in bank_divisors:
        for rgw in rank_divisors:
            cost = total_move_cost(sgw, bgw, rgw)
            if cost < min_cost:
                min_cost = cost
                best_config = (sgw, bgw, rgw)

print("OPTIMAL CONFIGURATION:")
print(f"  subarray_grid_width = {best_config[0]}")
print(f"  bank_grid_width = {best_config[1]}")
print(f"  rank_grid_width = {best_config[2]}")
print(f"  Total move cost = {min_cost:,.0f}")
print()

# Show top 10 configurations
print("Top 10 configurations:")
results = []
for sgw in subarray_divisors:
    for bgw in bank_divisors:
        for rgw in rank_divisors:
            cost = total_move_cost(sgw, bgw, rgw)
            results.append((cost, sgw, bgw, rgw))

results.sort()
for i, (cost, sgw, bgw, rgw) in enumerate(results[:10], 1):
    print(f"{i:2}. Cost={cost:12,.0f}  subarray={sgw:2}, bank={bgw:2}, rank={rgw:2}")
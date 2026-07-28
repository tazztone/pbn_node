import timeit


class MockColorPalette:
    def __init__(self, hex_colors):
        self.hex_colors = hex_colors

class MockGeneratorKeys:
    def group_paths_by_color(self, regions, colors, region_colors=None):
        grouped = {}
        for region_id in regions.keys():
            if region_colors and region_id in region_colors:
                color_idx = region_colors[region_id]
            else:
                color_idx = (region_id - 1) % len(colors.hex_colors)
            color_hex = colors.hex_colors[color_idx]
            if color_hex not in grouped:
                grouped[color_hex] = []
            grouped[color_hex].append(region_id)
        return grouped

class MockGeneratorNoKeys:
    def group_paths_by_color(self, regions, colors, region_colors=None):
        grouped = {}
        for region_id in regions:
            if region_colors and region_id in region_colors:
                color_idx = region_colors[region_id]
            else:
                color_idx = (region_id - 1) % len(colors.hex_colors)
            color_hex = colors.hex_colors[color_idx]
            if color_hex not in grouped:
                grouped[color_hex] = []
            grouped[color_hex].append(region_id)
        return grouped

regions = dict.fromkeys(range(100000))
colors = MockColorPalette(hex_colors=["#000000", "#FFFFFF", "#FF0000"])
region_colors = {i: i % 3 for i in range(100000)}

gen_keys = MockGeneratorKeys()
gen_no_keys = MockGeneratorNoKeys()

def bench_keys():
    gen_keys.group_paths_by_color(regions, colors, region_colors)

def bench_no_keys():
    gen_no_keys.group_paths_by_color(regions, colors, region_colors)

# Warmup
bench_keys()
bench_no_keys()

n = 100
time_keys = timeit.timeit(bench_keys, number=n)
time_no_keys = timeit.timeit(bench_no_keys, number=n)

print(f"Time with .keys(): {time_keys:.4f}s")
print(f"Time without .keys(): {time_no_keys:.4f}s")
print(f"Improvement: {(time_keys - time_no_keys) / time_keys * 100:.2f}%")

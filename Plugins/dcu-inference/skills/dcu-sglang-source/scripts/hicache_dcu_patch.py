#!/usr/bin/env python3
"""SGLang HiCache DCU 适配一键补丁（Bug B + Bug C，幂等）。

适用：SGLang 0.5.12+das.opt1.dtk2604 海光 fork，DCU 上跑 HiCache
（--enable-hierarchical-cache）混合 Mamba + 全注意力模型（qwen3_5 类）。

两个 bug：
  Bug B: memory_pool_host.py — direct 后端分页 KV 布局与 host 扁平布局不匹配
         （GPU 驱逐→写回 host 时 RuntimeError: size of tensor a (2) must match b (64)）
  Bug C: hi_mamba_radix_cache.py — load_back 回读路径 protect_host/release_host 缺失
         （L2 host 回读时 TypeError: NoneType has no len() / RuntimeError: counter already zero）

用法：
  python3 hicache_dcu_patch.py            # 打补丁（幂等，已打过则跳过）
  python3 hicache_dcu_patch.py --check    # 只检查是否已打，不改文件

⚠ 版本锚点：Bug C 的 4 个锚点（R1-R4）基于 0.5.12+das.opt1.dtk2604 的 load_back
原始代码。其他 SGLang 版本若 load_back 实现有差异，锚点 assert 会失败（不会静默
打错），需先核对 hi_mamba_radix_cache.py 的 load_back 源码再调整锚点。
"""
import sys
import shutil
import os

SG = "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache"
P_HOST = os.path.join(SG, "memory_pool_host.py")
P_MAMBA = os.path.join(SG, "hi_mamba_radix_cache.py")

CHECK_ONLY = "--check" in sys.argv


def backup(path):
    bak = path + ".bak_dcu"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup: {bak}")


def patch_bug_b():
    """Bug B: memory_pool_host.py direct 后端分页 KV 布局回退。"""
    print("[Bug B] memory_pool_host.py")
    if not os.path.exists(P_HOST):
        print("  SKIP: file not found")
        return
    s = open(P_HOST).read()
    if "_dcu_d2h_paged" in s:
        print("  already applied, skipping")
        return
    if CHECK_ONLY:
        print("  NOT applied")
        return
    backup(P_HOST)

    anchor = "    def load_to_device_per_layer("
    assert anchor in s, "anchor not found"
    methods = '''    # ---- DCU paged-layout fallback for the "direct" io backend ----
    # On DCU (SGLANG_KV_LAYOUT_DCU_FA=true) the device KV pool is paged:
    #   K[lid] = (page_num, head_num, page_size, head_dim)
    #   V[lid] = (page_num, head_num, v_head_dim, page_size)
    # while the host pool (layer_first) is flat: (size, head_num, head_dim).
    # The compiled transfer_kv_direct ops assume a flat token-indexed device
    # layout and fail with a 2-vs-64 broadcast error on the paged layout.
    # These pure-PyTorch gather/scatter helpers bridge paged<->flat. They only
    # run on the cold path (cache eviction D2H / load-back H2D).
    def _dcu_paged_device(self, device_pool):
        try:
            return device_pool.k_buffer[0].dim() == 4
        except Exception:
            return False

    def _dcu_d2h_paged(self, device_pool, host_indices, device_indices):
        ps = self.page_size
        dev = device_pool.k_buffer[0].device
        pg = (device_indices // ps).to(dev)
        off = (device_indices % ps).to(dev)
        hidx = host_indices.to(self.device)
        for lid in range(self.layer_num):
            self.k_data_refs[lid][hidx] = device_pool.k_buffer[lid][
                pg, :, off, :
            ].to(self.device)
            self.v_data_refs[lid][hidx] = device_pool.v_buffer[lid][
                pg, :, :, off
            ].to(self.device)

    def _dcu_h2d_paged(self, device_pool, host_indices, device_indices, layer_id):
        ps = self.page_size
        dev = device_pool.k_buffer[layer_id].device
        pg = (device_indices // ps).to(dev)
        off = (device_indices % ps).to(dev)
        hidx = host_indices.to(self.device)
        device_pool.k_buffer[layer_id][pg, :, off, :] = self.k_data_refs[layer_id][
            hidx
        ].to(dev)
        device_pool.v_buffer[layer_id][pg, :, :, off] = self.v_data_refs[layer_id][
            hidx
        ].to(dev)

'''
    s = s.replace(anchor, methods + anchor, 1)

    h2d_old = '''        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],'''
    h2d_new = '''        elif io_backend == "direct":
            if self.layout == "layer_first" and self._dcu_paged_device(device_pool):
                self._dcu_h2d_paged(
                    device_pool, host_indices, device_indices, layer_id
                )
                return
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],'''
    assert h2d_old in s, "H2D call site not found"
    s = s.replace(h2d_old, h2d_new, 1)

    d2h_old = '''        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,'''
    d2h_new = '''        elif io_backend == "direct":
            if self.layout == "layer_first" and self._dcu_paged_device(device_pool):
                self._dcu_d2h_paged(device_pool, host_indices, device_indices)
                return
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,'''
    assert d2h_old in s, "D2H call site not found"
    s = s.replace(d2h_old, d2h_new, 1)

    open(P_HOST, "w").write(s)
    print("  applied OK")


def patch_bug_c():
    """Bug C: hi_mamba_radix_cache.py load_back protect/release 缺失。"""
    print("[Bug C] hi_mamba_radix_cache.py")
    if not os.path.exists(P_MAMBA):
        print("  SKIP: file not found (纯注意力模型无此文件，正常)")
        return
    s = open(P_MAMBA).read()
    if "n.protect_host()" in s:
        print("  already applied, skipping")
        return
    if CHECK_ONLY:
        print("  NOT applied")
        return
    backup(P_MAMBA)

    # R1: 收集 nodes_to_load 后 protect_host
    r1_old = """        else:
            ancestor_node = node

        mamba_restore_nodes = []"""
    r1_new = """        else:
            ancestor_node = node

        for n in nodes_to_load:
            n.protect_host()

        mamba_restore_nodes = []"""
    assert r1_old in s, "R1 anchor not found"
    s = s.replace(r1_old, r1_new, 1)

    # R2: skip 路径 release
    r2_old = """            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancestor_node)
            return None"""
    r2_new = """            # skip loading back if the total size is too small or exceeding the memory quota
            for n in nodes_to_load:
                n.release_host()
            self.dec_lock_ref(ancestor_node)
            return None"""
    assert r2_old in s, "R2 anchor not found"
    s = s.replace(r2_old, r2_new, 1)

    # R3: None 分支 release
    r3_old = """        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None"""
    r3_new = """        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            return None"""
    assert r3_old in s, "R3 anchor not found"
    s = s.replace(r3_old, r3_new, 1)

    # R4: 成功路径 release
    r4_old = """        self._update_leaf_status(ancestor_node)

        self.inc_lock_ref(last_hit_node)"""
    r4_new = """        self._update_leaf_status(ancestor_node)

        for n in nodes_to_load:
            n.release_host()
        self.inc_lock_ref(last_hit_node)"""
    assert r4_old in s, "R4 anchor not found"
    s = s.replace(r4_old, r4_new, 1)

    open(P_MAMBA, "w").write(s)
    print("  applied OK")


def main():
    print(f"mode: {'CHECK ONLY' if CHECK_ONLY else 'APPLY'}")
    patch_bug_b()
    patch_bug_c()
    if not CHECK_ONLY:
        import py_compile
        for p in (P_HOST, P_MAMBA):
            if os.path.exists(p):
                py_compile.compile(p, doraise=True)
        print("py_compile OK")
    print("done")


if __name__ == "__main__":
    main()

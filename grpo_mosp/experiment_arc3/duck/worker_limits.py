"""Kernel restrictions applied inside Duck's Python tool subprocess."""

import ctypes
import errno
import importlib
import resource


def restrict_worker(safe_modules):
    # Load the tool's allowed standard-library modules before closing file access.
    for name in safe_modules:
        importlib.import_module(name)
    libc = ctypes.CDLL(None, use_errno=True)
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    memory_limit = 512 * 1024**2
    if hard_limit != resource.RLIM_INFINITY:
        memory_limit = min(memory_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

    # Landlock is available on the x86_64 Linux evaluation host. An empty
    # ruleset denies file access for all handled filesystem rights.
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3 or libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("Duck tool worker requires Linux Landlock ABI 3 or newer")
    handled = ctypes.c_uint64((1 << (16 if abi >= 5 else 15)) - 1)
    ruleset = libc.syscall(444, ctypes.byref(handled), ctypes.sizeof(handled), 0)
    if ruleset < 0:
        raise RuntimeError("Cannot create Duck tool filesystem restrictions")
    try:
        if libc.syscall(446, ruleset, 0) != 0:
            raise RuntimeError("Cannot apply Duck tool filesystem restrictions")
    finally:
        libc.close(ruleset)

    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    policy = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not policy:
        raise RuntimeError("Cannot create Duck tool syscall restrictions")
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO
        for name in (
            "socket", "socketpair", "execve", "execveat", "clone", "clone3", "fork", "vfork",
            "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd",
            "kill", "tkill", "tgkill", "pidfd_send_signal", "io_uring_setup",
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0 or seccomp.seccomp_rule_add(policy, deny, number, 0) != 0:
                raise RuntimeError(f"Cannot restrict Duck tool syscall {name}")
        if seccomp.seccomp_load(policy) != 0:
            raise RuntimeError("Cannot apply Duck tool syscall restrictions")
    finally:
        seccomp.seccomp_release(policy)

"""
Tests for Group 1: LamExpl, LamImpl, AlphasLam, AlphasExact.

Benchmark values were generated with rundec-python 0.7 (the CRunDec C++ wrapper),
which is a test-only dependency and not required by rundex itself.

Generation script (run once to produce the constants below):

    import rundec
    c = rundec.CRunDec()
    c.LamExpl(0.118, 91.18, 5, <loops>)   # etc.

Physical inputs:
    αs(MZ) = 0.118,  MZ = 91.18 GeV,  nf = 5
"""

import pytest
from rundex import AlphasExact, AlphasLam, LamExpl, LamImpl

# Standard inputs
AS_MZ = 0.118
MZ = 91.18
NF = 5

# ---------------------------------------------------------------------------
# Reference values from rundec-python 0.7 (CRunDec C++ wrapper)
# Generated via: c = rundec.CRunDec(); c.<func>(<args>)
# ---------------------------------------------------------------------------

# c.LamExpl(0.118, 91.18, 5, loops)
REF_LAMEXPL = {
    1: 0.08781976022692606,
    2: 0.20866950236134246,
    3: 0.20890477421520084,
    4: 0.20859280429615878,
    5: 0.20860682795985377,
}

# c.LamImpl(0.118, 91.18, 5, loops)
REF_LAMIMPL = {
    1: 0.08781976860207104,
    2: 0.22621486792994805,
    3: 0.20834830299127508,
    4: 0.20834647971890383,
    5: 0.2086792332552601,
}

# c.AlphasLam(c.LamExpl(0.118, 91.18, 5, loops), 91.18, 5, loops)
# Note: AlphasLam(LamExpl(...)) is NOT exactly αs — the two expansions
# are asymptotically consistent but not term-by-term inverse at finite order.
REF_ALPHASLAMEXPL_ROUNDTRIP = {
    1: 0.118,
    2: 0.11657945907728862,
    3: 0.1180475622582365,
    4: 0.11802109418251719,
    5: 0.11799381078864873,
}

# c.AlphasExact(0.118, 91.18, 50.0, 5, loops)
REF_ALPHASEXACT_50 = {
    1: 0.12917439914315504,
    2: 0.129784580079333,
    3: 0.12981291235774434,
    4: 0.1298203686788109,
    5: 0.1298206045557886,
}

# c.AlphasExact(0.118, 91.18, 10.0, 5, loops)
REF_ALPHASEXACT_10 = {
    1: 0.17308058936475007,
    2: 0.17787083454631622,
    3: 0.1781418579967483,
    4: 0.17822724774207974,
    5: 0.17823051657011182,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def rel_err(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# LamExpl
# ---------------------------------------------------------------------------

class TestLamExpl:

    @pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
    def test_vs_crundec(self, loops):
        """Match CRunDec reference to 10 significant digits."""
        lam = LamExpl(AS_MZ, MZ, NF, loops)
        assert rel_err(lam, REF_LAMEXPL[loops]) < 1e-10, (
            f"loops={loops}: got {lam:.10e}, ref {REF_LAMEXPL[loops]:.10e}"
        )

    def test_invalid_loops(self):
        with pytest.raises(ValueError):
            LamExpl(AS_MZ, MZ, NF, 0)
        with pytest.raises(ValueError):
            LamExpl(AS_MZ, MZ, NF, 6)


# ---------------------------------------------------------------------------
# LamImpl
# ---------------------------------------------------------------------------

class TestLamImpl:

    @pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
    def test_vs_crundec(self, loops):
        """Match CRunDec reference to 7 significant digits.

        LamImpl uses numerical root-finding; tiny differences (~1e-7 relative)
        from CRunDec's own root-finder are expected and acceptable.
        """
        lam = LamImpl(AS_MZ, MZ, NF, loops)
        assert rel_err(lam, REF_LAMIMPL[loops]) < 1e-7, (
            f"loops={loops}: got {lam:.10e}, ref {REF_LAMIMPL[loops]:.10e}"
        )

    def test_invalid_loops(self):
        with pytest.raises(ValueError):
            LamImpl(AS_MZ, MZ, NF, 6)


# ---------------------------------------------------------------------------
# AlphasLam
# ---------------------------------------------------------------------------

class TestAlphasLam:

    @pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
    def test_roundtrip_vs_crundec(self, loops):
        """
        AlphasLam(LamExpl(αs)) is NOT exactly αs at finite loop order.
        Check we match the CRunDec roundtrip value.
        """
        lam = LamExpl(AS_MZ, MZ, NF, loops)
        as_back = AlphasLam(lam, MZ, NF, loops)
        ref = REF_ALPHASLAMEXPL_ROUNDTRIP[loops]
        assert rel_err(as_back, ref) < 1e-10, (
            f"loops={loops}: got {as_back:.10e}, ref {ref:.10e}"
        )

    def test_running_direction(self):
        """αs should decrease when μ increases (asymptotic freedom)."""
        lam = LamExpl(AS_MZ, MZ, NF, 4)
        as_low = AlphasLam(lam, 50.0, NF, 4)
        as_high = AlphasLam(lam, 200.0, NF, 4)
        assert as_low > as_high

    def test_invalid_loops(self):
        lam = LamExpl(AS_MZ, MZ, NF, 4)
        with pytest.raises(ValueError):
            AlphasLam(lam, MZ, NF, 6)


# ---------------------------------------------------------------------------
# AlphasExact
# ---------------------------------------------------------------------------

class TestAlphasExact:

    def test_identity_at_same_scale(self):
        result = AlphasExact(AS_MZ, MZ, MZ, NF, 4)
        assert result == AS_MZ

    @pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
    def test_vs_crundec_mu50(self, loops):
        """Match CRunDec at μ=50 GeV to 8 significant digits."""
        result = AlphasExact(AS_MZ, MZ, 50.0, NF, loops)
        ref = REF_ALPHASEXACT_50[loops]
        assert rel_err(result, ref) < 1e-8, (
            f"loops={loops}: got {result:.10e}, ref {ref:.10e}"
        )

    @pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
    def test_vs_crundec_mu10(self, loops):
        """Match CRunDec at μ=10 GeV to 8 significant digits."""
        result = AlphasExact(AS_MZ, MZ, 10.0, NF, loops)
        ref = REF_ALPHASEXACT_10[loops]
        assert rel_err(result, ref) < 1e-8, (
            f"loops={loops}: got {result:.10e}, ref {ref:.10e}"
        )

    def test_round_trip(self):
        """Run up then back down — should recover the starting value to 9 digits."""
        mu_high = 500.0
        as_up = AlphasExact(AS_MZ, MZ, mu_high, NF, 5)
        as_down = AlphasExact(as_up, mu_high, MZ, NF, 5)
        assert rel_err(as_down, AS_MZ) < 1e-8

    def test_asymptotic_freedom(self):
        """αs must decrease as μ increases."""
        as_low = AlphasExact(AS_MZ, MZ, 50.0, NF, 4)
        as_high = AlphasExact(AS_MZ, MZ, 500.0, NF, 4)
        assert as_low > AS_MZ > as_high

    def test_loop_convergence(self):
        """Higher-loop results should converge (differences shrink)."""
        mu_test = 10.0
        results = [AlphasExact(AS_MZ, MZ, mu_test, NF, l) for l in range(1, 6)]
        diffs = [abs(results[i + 1] - results[i]) for i in range(4)]
        for i in range(3):
            assert diffs[i + 1] < diffs[i], (
                f"Loop convergence failed: Δ({i+2},{i+3}) = {diffs[i+1]:.2e} "
                f"> Δ({i+1},{i+2}) = {diffs[i]:.2e}"
            )

    def test_invalid_loops(self):
        with pytest.raises(ValueError):
            AlphasExact(AS_MZ, MZ, 50.0, NF, 0)
        with pytest.raises(ValueError):
            AlphasExact(AS_MZ, MZ, 50.0, NF, 6)



import pytest

from numpy.testing import assert_approx_equal
from pyadjoint import *


def test_lcoe():
    energy = [AdjFloat(2.0), AdjFloat(2.5)]
    cost = [AdjFloat(13.0), AdjFloat(7.0)]

    discount_rate = 0.05

    discounted_cost = [0] * len(cost)
    discounted_energy = [0] * len(energy)

    for n in range(0, len(cost)):
        discounted_cost[n] = (cost[n] / ((1 + discount_rate) ** n))
        discounted_energy[n] = (energy[n] / ((1 + discount_rate) ** n))

    lcoe = sum(discounted_cost) / sum(discounted_energy)

    assert_approx_equal(lcoe, 4.4891304347826075)

    # d L /d e0
    rf = ReducedFunctional(lcoe, Control(energy[0]))
    assert rf(AdjFloat(2.0)) == 4.4891304347826075  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(2.0)), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(energy[0]), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), -1.0246928166351605)  # -(13+7/1.05)/(2.0+2.5/1.05)**2
    assert_approx_equal(rf(AdjFloat(3.0)), 3.6548672566371674)  # (13+7/1.05)/(3.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), -0.6792231184900931)  # -(13+7/1.05)/(3.0+2.5/1.05)**2
    assert_approx_equal(rf(AdjFloat(2.0)), 4.4891304347826075)  # to set value back for next test

    # d L /d e1
    rf = ReducedFunctional(lcoe, Control(energy[1]))
    assert rf(AdjFloat(2.5)) == 4.4891304347826075  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(2.5)), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(energy[1]), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), -0.9758979206049146)  # -(13+7/1.05)/(1.05*(2.0+2.5/1.05)**2)
    assert_approx_equal(rf(AdjFloat(3.0)), 4.049019607843136)  # (13+7/1.05)/(2.0+3.0/1.05)
    assert_approx_equal(rf.derivative(), -0.7939254133025757)  # -(13+7/1.05)/(1.05*(2.0+3.0/1.05)**2)
    assert_approx_equal(rf(AdjFloat(2.5)), 4.4891304347826075)  # to set value back for next test

    # d L /d c0
    rf = ReducedFunctional(lcoe, Control(cost[0]))
    assert rf(AdjFloat(13.0)) == 4.4891304347826075  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(13.0)), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(cost[0]), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), 0.22826086956521738)  # 1/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(15.0)), 4.945652173913042)  # (15+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), 0.22826086956521738)  # 1/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(13.0)), 4.4891304347826075)  # to set value back for next test

    # d L /d c1
    rf = ReducedFunctional(lcoe, Control(cost[1]))
    assert rf(AdjFloat(7.0)) == 4.4891304347826075  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(7.0)), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(cost[1]), 4.4891304347826075)  # (13+7/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), 0.21739130434782605)  # (1/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(9.0)), 4.92391304347826)  # (13+9/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf.derivative(), 0.21739130434782605)  # (1/1.05)/(2.0+2.5/1.05)
    assert_approx_equal(rf(AdjFloat(7.0)), 4.4891304347826075)  # to set value back for next test

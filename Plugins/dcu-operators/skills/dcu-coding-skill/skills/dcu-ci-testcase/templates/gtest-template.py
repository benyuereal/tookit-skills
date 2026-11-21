"""Template snippet for C++ gtest style tests."""

GTEST_TEMPLATE = r"""
#include <gtest/gtest.h>

TEST(HipKernelTest, BasicBehavior) {
    EXPECT_TRUE(true);
}
"""

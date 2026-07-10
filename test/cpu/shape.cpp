#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include <vext/shape.hpp>

namespace
{

void
expect_shape(
	const vext::Shape&                         shape,
	const std::initializer_list<std::uint32_t> dims,
	const std::initializer_list<std::uint32_t> strides,
	const std::uint32_t                        length)
{
	EXPECT_EQ(shape.dims(), std::vector<std::uint32_t>(dims));
	EXPECT_EQ(shape.strides(), std::vector<std::uint32_t>(strides));
	EXPECT_EQ(shape.size(), dims.size());
	EXPECT_EQ(shape.length(), length);
}

}

TEST(ShapeCpu, ConstructsFromVariadicDimensions)
{
	const vext::Shape shape(2, 3, 4);

	expect_shape(shape, { 2, 3, 4 }, { 12, 4, 1 }, 24);
}

TEST(ShapeCpu, ConstructsFromCopiedAndMovedVectors)
{
	const std::vector<std::uint32_t> dims{ 5, 6 };
	const vext::Shape                copied(dims);
	vext::Shape                      moved(std::vector<std::uint32_t>{ 7, 8, 9 });

	expect_shape(copied, { 5, 6 }, { 6, 1 }, 30);
	expect_shape(moved, { 7, 8, 9 }, { 72, 9, 1 }, 504);
}

TEST(ShapeCpu, SupportsPositiveAndNegativeIndexing)
{
	const vext::Shape shape(2, 3, 4);

	EXPECT_EQ(shape[0], 2);
	EXPECT_EQ(shape[2], 4);
	EXPECT_EQ(shape[-1], 4);
	EXPECT_EQ(shape[-2], 3);
	EXPECT_EQ(shape[-3], 2);
	EXPECT_EQ(shape.at(1), 3);
	EXPECT_EQ(shape.at(-2), 3);
	EXPECT_THROW((void)shape.at(3), std::out_of_range);
	EXPECT_THROW((void)shape.at(-3), std::out_of_range) << "Current implementation rejects abs(index) >= rank, so -rank is not accepted by at().";
}

TEST(ShapeCpu, IteratesDimensions)
{
	const vext::Shape                shape(3, 1, 2);
	const std::vector<std::uint32_t> dims(shape.begin(), shape.end());

	EXPECT_EQ(dims, (std::vector<std::uint32_t>{ 3, 1, 2 }));
}

TEST(ShapeCpu, ComparesByDimensions)
{
	EXPECT_EQ(vext::Shape(2, 3), vext::Shape(2, 3));
	EXPECT_NE(vext::Shape(2, 3), vext::Shape(3, 2));
}

TEST(ShapeCpu, RejectsInvalidRuntimeRankAndOverflow)
{
	const std::vector<std::uint32_t> empty;
	EXPECT_THROW({ const vext::Shape shape(empty); }, std::runtime_error);

	std::vector<std::uint32_t> too_many(33, 1);
	EXPECT_THROW({ const vext::Shape shape(too_many); }, std::runtime_error);

	const std::vector<std::uint32_t> overflowing{ std::numeric_limits<std::uint32_t>::max(), 2 };
	EXPECT_THROW({ const vext::Shape shape(overflowing); }, std::overflow_error);
}

TEST(ShapeCpu, BroadcastEmbedsTargetShapeAndZeroesBroadcastStrides)
{
	const vext::Shape broadcast = vext::Shape::broadcast(vext::Shape(2, 3), vext::Shape(3));

	expect_shape(broadcast, { 1, 3 }, { 0, 1 }, 3);
}

TEST(ShapeCpu, BroadcastSupportsScalarLikeLengthOneTarget)
{
	const vext::Shape broadcast = vext::Shape::broadcast(vext::Shape(2, 3, 4), vext::Shape(1));

	expect_shape(broadcast, { 1, 1, 1 }, { 0, 0, 0 }, 1);
}

TEST(ShapeCpu, BroadcastRejectsIncompatibleShapes)
{
	EXPECT_THROW(vext::Shape::broadcast(vext::Shape(2, 3), vext::Shape(4)), std::runtime_error);
	EXPECT_THROW(vext::Shape::broadcast(vext::Shape(2), vext::Shape(2, 1)), std::runtime_error);
}

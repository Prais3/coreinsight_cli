# bad_loop.py

def count_unique_numbers(numbers):
    """
    Inefficiently counts unique numbers in a list.
    This implementation is intentionally unoptimized.
    """
    unique = []

    for i in range(len(numbers)):
        already_exists = False

        for j in range(len(unique)):
            if numbers[i] == unique[j]:
                already_exists = True
                break

        if not already_exists:
            unique.append(numbers[i])

    return len(unique)


def slow_sum_of_squares(n):
    """
    Intentionally slow calculation of sum of squares up to n.
    """
    total = 0
    for i in range(n):
        square = 0
        for _ in range(i):
            square += i  # Repeated addition instead of i * i
        total += square
    return total


if __name__ == "__main__":
    nums = [1, 2, 2, 3, 4, 4, 5, 1, 6, 7, 7, 8, 9, 9]
    print("Unique count:", count_unique_numbers(nums))
    print("Slow sum of squares:", slow_sum_of_squares(1000))
#include <cmath>
#include <vector>

// Very slow prime check (O(n))
bool isPrimeSlow(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i < n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// Very slow recursive factorial (no memoization)
long double factorialSlow(int n) {
    if (n <= 1) return 1.0;
    return n * factorialSlow(n - 1);
}

// Intentionally slow computational math function
long double slowComputation(int limit) {
    long double result = 0.0;

    for (int i = 1; i <= limit; ++i) {
        for (int j = 1; j <= limit; ++j) {
            long double val = std::sin(i) * std::cos(j);
            val += std::pow(i, 1.5) / (j + 1);

            if (isPrimeSlow(i * j)) {
                val *= factorialSlow((i % 10) + 5);
            }

            result += std::sqrt(std::fabs(val));
        }
    }

    return result;
}
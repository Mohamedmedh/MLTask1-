# Function to compute matrix multiplication
def matrix_multiply(A, B):
    """Multiply two matrices A and B."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions do not match for multiplication.")

    # Initialize result matrix with zeros
    result = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))

    return result


# Function to compute the transpose of a matrix
def transpose_matrix(A):
    """Return the transpose of a matrix A."""
    rows, cols = len(A), len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]


# Function to compute the inverse of a square matrix using Gaussian elimination
def inverse_matrix(A):
    """Compute the inverse of a square matrix A using Gaussian elimination."""
    n = len(A)

    # Create an augmented matrix [A | I]
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    A = [row[:] + I_row[:] for row, I_row in zip(A, I)]

    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        diag = A[i][i]
        if diag == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        for j in range(2 * n):
            A[i][j] /= diag
        # Make other elements in the column 0
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(2 * n):
                    A[k][j] -= factor * A[i][j]

    # Extract the inverse matrix
    return [row[n:] for row in A]


# Function to solve for theta using the Normal Equation
def normal_equation(X, y):
    """Compute theta using the normal equation: theta = (X^T X)^(-1) X^T y."""
    X_T = transpose_matrix(X)  # Compute X^T
    X_T_X = matrix_multiply(X_T, X)  # Compute X^T X
    X_T_y = matrix_multiply(X_T, y)  # Compute X^T y
    X_T_X_inv = inverse_matrix(X_T_X)  # Compute (X^T X)^(-1)
    theta = matrix_multiply(X_T_X_inv, X_T_y)  # Compute theta
    return theta


# Dataset: House Price = f(square footage, bedrooms, bathrooms)
data = [
    [1, 1400, 3, 2, 245000],  # [Bias term, Square Feet, Bedrooms, Bathrooms, Price]
    [1, 1600, 3, 3, 312000],
    [1, 1700, 4, 2, 279000],
    [1, 1875, 4, 3, 308000],
    [1, 1100, 2, 1, 199000]
]

# Extract features (X) and target (y)
X = [row[:-1] for row in data]  # Feature matrix (including bias term)
y = [[row[-1]] for row in data]  # Target vector reshaped to column matrix

# Compute theta
theta = normal_equation(X, y)

# Print the learned parameters
print("Theta values (coefficients):")
for i, coef in enumerate(theta):
    print(f"θ{i} = {coef[0]:.4f}")

# Predict price of a new house [1, 1550 sq ft, 3 beds, 2 baths]
new_house = [[1, 1550, 3, 2]]
predicted_price = matrix_multiply(new_house, theta)
print(f"\nPredicted price for a 1550 sq ft house with 3 beds and 2 baths: ${predicted_price[0][0]:,.2f}")

import streamlit as st
import numpy as np


def parse_matrix_input(matrix_str):
    """
    Parse a string input into a NumPy matrix.
    Expected format: rows separated by semicolons, elements by spaces or commas.
    Example: "1 2; 3 4" for [[1,2],[3,4]]
    """
    try:
        rows = [row.strip() for row in matrix_str.split(';') if row.strip()]
        matrix = []
        for row in rows:
            elements = [float(x.strip()) for x in row.replace(',', ' ').split() if x.strip()]
            matrix.append(elements)
        matrix = np.array(matrix)
        if len(set(len(row) for row in matrix)) > 1:
            raise ValueError("All rows must have the same number of columns.")
        return matrix
    except Exception as e:
        raise ValueError(f"Invalid matrix format: {e}")


def main():
    st.title("Matrix Operations Tool")

    # Sidebar for operation selection
    operation = st.sidebar.selectbox(
        "Select Operation",
        ["Addition", "Subtraction", "Multiplication", "Transpose", "Determinant"]
    )

    # Input sections
    if operation in ["Addition", "Subtraction", "Multiplication"]:
        st.header("Matrix A")
        matrix_a_input = st.text_area("Enter Matrix A (rows separated by ';', elements by spaces/commas):")

        st.header("Matrix B")
        matrix_b_input = st.text_area("Enter Matrix B (rows separated by ';', elements by spaces/commas):")

        if st.button("Compute"):
            try:
                A = parse_matrix_input(matrix_a_input)
                B = parse_matrix_input(matrix_b_input)

                if operation == "Addition":
                    if A.shape != B.shape:
                        st.error("Matrices must have the same dimensions for addition.")
                    else:
                        result = A + B
                        st.subheader("Result (A + B):")
                        st.write(result)

                elif operation == "Subtraction":
                    if A.shape != B.shape:
                        st.error("Matrices must have the same dimensions for subtraction.")
                    else:
                        result = A - B
                        st.subheader("Result (A - B):")
                        st.write(result)

                elif operation == "Multiplication":
                    if A.shape[1] != B.shape[0]:
                        st.error("Number of columns in A must equal number of rows in B for multiplication.")
                    else:
                        result = np.dot(A, B)
                        st.subheader("Result (A * B):")
                        st.write(result)
            except ValueError as e:
                st.error(e)

    else:
        st.header("Matrix A")
        matrix_a_input = st.text_area("Enter Matrix A (rows separated by ';', elements by spaces/commas):")

        if st.button("Compute"):
            try:
                A = parse_matrix_input(matrix_a_input)

                if operation == "Transpose":
                    result = A.T
                    st.subheader("Result (Transpose of A):")
                    st.write(result)

                elif operation == "Determinant":
                    if A.shape[0] != A.shape[1]:
                        st.error("Matrix must be square for determinant calculation.")
                    else:
                        result = np.linalg.det(A)
                        st.subheader("Determinant of A:")
                        st.write(f"{result:.4f}")
            except ValueError as e:
                st.error(e)

    # Example input format
    st.sidebar.markdown("### Input Format Example")
    st.sidebar.text("For a 2x2 matrix:\n1 2; 3 4\n(or 1,2;3,4)")


if __name__ == "__main__":
    main()

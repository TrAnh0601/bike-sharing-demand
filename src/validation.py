import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import BaseCrossValidator

# v1.5 Expanding Window with Gap
class ExpandingWindowSplitter(BaseCrossValidator):
    """
    Expanding Window Cross-Validator with Gap.
    """
    def __init__(self, n_splits=5, gap_size=0, test_size=None, min_train_size=None):
        self.n_splits = n_splits
        self.gap_size = gap_size  # Số giờ/ngày cách ly giữa Train và Test
        self.test_size = test_size  # Kích thước tập Test (ví dụ: 1 tuần = 168)
        self.min_train_size = min_train_size  # Kích thước train tối thiểu

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Nếu không set test_size, tự động chia đều phần đuôi dữ liệu
        if self.test_size is None:
            # Dành 20% dữ liệu cuối cùng để chia thành n_splits fold
            validation_start = int(n_samples * 0.8)
            # Mỗi fold test sẽ có kích thước bằng nhau
            self.test_size = int((n_samples - validation_start) / self.n_splits)

        # Duyệt ngược từ Fold cuối cùng về Fold đầu tiên
        for i in range(self.n_splits):
            # Tính toán các điểm cắt (Indices)
            # Ví dụ: n_splits=5, i chạy từ 0 đến 4
            # i=0: Fold cuối cùng (gần hiện tại nhất)
            # i=4: Fold xa nhất (ít dữ liệu train nhất)

            # Để logic thuận tự nhiên (Fold 1 là ít data nhất), ta đảo index:
            fold_idx = self.n_splits - 1 - i

            test_end = n_samples - fold_idx * self.test_size
            test_start = test_end - self.test_size

            train_end = test_start - self.gap_size

            # Kiểm tra ràng buộc min_train_size
            if self.min_train_size and train_end < self.min_train_size:
                raise ValueError(
                    f"Fold {i + 1}: Not enough data for training ({train_end} samples). Reduce n_splits or min_train_size.")

            if train_end <= 0:
                raise ValueError(f"Fold {i + 1}: Gap/Test size too large, train set is empty.")

            train_index = indices[:train_end]
            test_index = indices[test_start:test_end]

            yield train_index, test_index

    def plot_splits(self, X, y=None):
        """Hàm trực quan hóa các Fold để kiểm tra Gap"""
        plt.figure(figsize=(12, 6))
        for i, (train_idx, test_idx) in enumerate(self.split(X, y)):
            # Vẽ Train
            plt.scatter([i] * len(train_idx), train_idx, c='blue', marker='_', lw=8, label='Train' if i == 0 else "")
            # Vẽ Test
            plt.scatter([i] * len(test_idx), test_idx, c='red', marker='_', lw=8, label='Test' if i == 0 else "")
            # Khoảng trắng ở giữa chính là Gap

        plt.legend()
        plt.title(f'Expanding Window Validation with Gap={self.gap_size}')
        plt.ylabel('Sample Index (Time)')
        plt.xlabel('CV Iteration')
        plt.grid(True, axis='x')
        plt.show()

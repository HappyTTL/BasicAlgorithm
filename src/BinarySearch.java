import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Tingting on 10/4/16.
 */
public class BinarySearch {
    /**
     * Median of Two Sorted Arrays
     */
    public double findMedianSortedArrays(int[] num1, int[] num2) {
        if (num1 == null || num1.length == 0) {
            return findMedianSingle(num2);
        }
        if (num2 == null || num2.length == 0) {
            return findMedianSingle(num1);
        }
        int m = num1.length, n = num2.length;
        if ((m + n) % 2 == 0) {
            return (findKth(num1, 0, num2, 0, (m + n) / 2) + findKth(num1, 0, num2, 0, (m + n) / 2 + 1)) / 2.0;
        } else {
            return findKth(num1, 0, num2, 0, (m + n) / 2 + 1);
        }
    }
    private double findMedianSingle(int[] num) {
        if (num == null || num.length == 0) {
            return 0;
        }
        return (num.length % 2 == 0) ? (num[(num.length - 1) / 2] + num[(num.length - 1) / 2 + 1]) / 2 : num[(num.length - 1) / 2];
    }
    private double findKth(int[] num1, int start1, int[] num2, int start2, int k) {
        if (start1 >= num1.length) {
            return (double) num2[start2 + k - 1];
        }
        if (start2 >= num2.length) {
            return (double) num1[start1 + k - 1];
        }
        if (k == 1) {
            return Math.min(num1[start1], num2[start2]);
        }
        int mid1 = start1 + k / 2 - 1, mid2 = start2 + k / 2 - 1;
        int value1 = mid1 >= num1.length ? Integer.MAX_VALUE : num1[mid1];
        int value2 = mid2 >= num2.length ? Integer.MAX_VALUE : num2[mid2];
        if (value1 > value2) {
            return findKth(num1, start1, num2, mid2 + 1, k - k / 2);
        } else {
            return findKth(num1, mid1 + 1, num2, start2, k - k / 2);
        }
    }

    /**
     * Pow(x,n)
     * @param args
     */
    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            return (1 / pow(x, n));
        }
        return pow(x, n);
    }
    private double pow(double x, int n) {
        if (n == 0) {
            return 1.0;
        }
        double v = pow(x, n / 2);
        if (n % 2 == 0) {
            return v * v;
        } else {
            return v * v * x;
        }
    }

    /**
     * Find Miniumum in Rotated Sorted Array: find first item less than nums[end].
     * @param args
     */
    public int findMin(int[] nums) {
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] > nums[end]) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (nums[start] < nums[end]) {
            return nums[start];
        } else {
            return nums[end];
        }
    }

    /**
     * Find Minimum in Rotated Sorted Array II: What if duplicates are allowed?

     Would this affect the run-time complexity? How and why?
     Worst Case scenario: 111111110, will be O(n); Either write a for loop or still use binary search.
     */
    public int findMinII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0;
        int end = nums.length - 1;
        while(start + 1 < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == nums[end]) {
                end--;
            } else if (nums[mid] < nums[end]) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (nums[start] <= nums[end]) {
            return nums[start];
        }
        return nums[end];
    }

    /**
     * Search in Rotated Sorted Array: when deal with sorted array, always use mid to compare with end
     * @param args
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        return searchRotatedArray(nums, target, 0, nums.length - 1);
    }
    private int searchRotatedArray(int[] nums, int target, int start, int end) {
        if (start > end) {
            return -1;
        }
        int mid = start + (end - start) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[mid] < nums[end]) {
            if (nums[mid] < target && target <= nums[end]) {
                return searchRotatedArray(nums, target, mid + 1, end);
            } else {
                return searchRotatedArray(nums, target, start, mid - 1);
            }
        } else {
            if (nums[start] <= target && target < nums[mid]) {
                return searchRotatedArray(nums, target, start, mid - 1);
            } else {
                return searchRotatedArray(nums, target, mid + 1, end);
            }
        }
    }

    /**
     * Search in Rotated Sorted Array II: What if duplicates are allowed?

     Would this affect the run-time complexity? How and why?
     Worst case scenario: 1111112, to find 2, time complexity will be O(n), so just write a iteration from 0 to n - 1
     */
    public boolean search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                return true;
            }
        }
        return false;
    }
    /**
     * Divide Two Integers: Divide two integers without using multiplication, division and mod operator.

     If it is overflow, return MAX_INT.
     * @param args
     */
    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return dividend >= 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        if (dividend == 0) {
            return 0;
        }
        //abs(INT_MIN)的值还是INT_MIN
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        boolean isNegative = (dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0);
        long d = Math.abs((long) dividend), r = Math.abs((long) divisor);
        int result = 0;
        while (d >= r) {
            int shift = 0;
            while (d >= (r << shift)) {
                shift++;
            }
            d -= r << (shift - 1);
            result += 1 << (shift - 1);
        }
        if (result > Integer.MAX_VALUE) {
            return Integer.MAX_VALUE;
        } else {
            return isNegative ? -result : result;
        }
    }

    /**
     * Sqrt(x);
     * @param args
     */
    public int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }
        long start = 1, end = x;
        while (start + 1 < end) {
            long mid = (long) (start + end) / 2;
            if (mid * mid == x) {
                return (int) mid;
            } else if (mid * mid < x){
                start = mid;
            } else {
                end = mid;
            }
        }
        if (end * end <= x) {
            return (int) end;
        } else {
            return (int) start;
        }
    }

    /* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */
    /** First Bad Version:
    class Solution extends VersionControl {
        public int firstBadVersion(int n) {
            int start = 1, end = n;
            while (start + 1 < end) {
                int mid = start + (end - start) / 2;
                if (isBadVersion(mid)) {
                    end = mid;
                } else {
                    start = mid;
                }
            }
            if (isBadVersion(start)) {
                return start;
            } else {
                return end;
            }
        }
    }
     */

    /** Find Peak Element: A peak element is an element that is greater than its neighbors.

     Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.

     The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

     You may imagine that num[-1] = num[n] = -∞.

     For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.

     This is to find an item which is greater than both left and right neighbour, if nums[mid] > nums[mid -1], there must
     be a peak value on the right, otherwise search the left.
     */

    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] > nums[mid - 1]) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (nums[start] > nums[end]) {
            return start;
        } else {
            return end;
        }
    }

    /**
     * Search for a range: Given a sorted array of integers, find the starting and ending position of a given target value.

     Your algorithm's runtime complexity must be in the order of O(log n).

     If the target is not found in the array, return [-1, -1].

     For example,
     Given [5, 7, 7, 8, 8, 10] and target value 8,
     return [3, 4].
     * @param args
     */
    public int[] searchRange(int[] nums, int target) {
        // do two binary search, one from left, one from right
        int[] result = new int[]{-1, -1};
        if (nums == null || nums.length == 0) {
            return result;
        }
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (nums[start] == target) {
            result[0] = start;
        } else if (nums[end] == target) {
            result[0] = end;
        }
        start = 0;
        end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] <= target) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (nums[end] == target) {
            result[1] = end;
        } else if (nums[start] == target) {
            result[1] = start;
        }
        return result;
    }

    /**
     * Search Insert Position: Given a sorted array and a target value, return the index if the target is found. If not,
     * return the index where it would be if it were inserted in order.

     You may assume no duplicates in the array.


     * @param args
     */
    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (nums[start] >= target) {
            return start;
        } else if (nums[end] >= target) {
            return end;
        } else {
            return end + 1;
        }
    }

    /** Find the Duplicate Number: Given an array nums containing n + 1 integers where each integer is between 1 and n
     * (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number,
     * find the duplicate one.

     Note:
     You must not modify the array (assume the array is read only).
     You must use only constant, O(1) extra space.
     Your runtime complexity should be less than O(n2).
     There is only one duplicate number in the array, but it could be repeated more than once.
     *
     */
    public int findDuplicate(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return -1;
        }
        int low = 1, high = nums.length - 1, left = 0;
        while (low + 1 < high) {
            int mid = low + (high - low) / 2;
            left = 0;
            for (int num : nums) {
                if (num <= mid) {
                    left++;
                }
            }
            if (left > mid) {
                high = mid;
            } else {
                low = mid;
            }
        }
        int mid = low + (high - low) / 2;
        left = 0;
        for (int num : nums) {
            if (num <= mid) {
                left++;
            }
        }
        if (left > mid) {
            return low;
        } else {
            return high;
        }
    }
    /**
     * Search a 2D Matrix: Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has
     * the following properties:

     Integers in each row are sorted from left to right.
     The first integer of each row is greater than the last integer of the previous row.
     For example,

     Consider the following matrix:

     [
     [1,   3,  5,  7],
     [10, 11, 16, 20],
     [23, 30, 34, 50]
     ]
     Given target = 3, return true.
     * @param args
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int m = matrix.length, n = matrix[0].length;
        int start = 0, end = m * n - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (matrix[mid / n][mid % n] == target) {
                return true;
            } else if (matrix[mid / n][mid % n] > target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (matrix[start / n][start % n] == target) {
            return true;
        }
        if (matrix[end / n][end % n] == target) {
            return true;
        }
        return false;
    }

    /**
     * Search a 2D Matrix II: Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

     Integers in each row are sorted in ascending from left to right.
     Integers in each column are sorted in ascending from top to bottom.
     For example,

     Consider the following matrix:

     [
     [1,   4,  7, 11, 15],
     [2,   5,  8, 12, 19],
     [3,   6,  9, 16, 22],
     [10, 13, 14, 17, 24],
     [18, 21, 23, 26, 30]
     ]
     Given target = 5, return true.

     Given target = 20, return false.
     * @param args
     * Basic Idea: search from right up corner or left down corner.
     */

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] < target) {
                row++;
            } else {
                col--;
            }
        }
        return false;
    }

    /**
     * Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest
     * element in the matrix.

     Note that it is the kth smallest element in the sorted order, not the kth distinct element.

     Example:

     matrix = [
     [ 1,  5,  9],
     [10, 11, 13],
     [12, 13, 15]
     ],
     k = 8,

     return 13.
     */

    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int min = matrix[0][0], max = matrix[n - 1][n - 1];
        while (min < max) {
            int mid = min + (max - min) / 2;
            int temp = count(matrix, n, mid);
            if (temp < k) {
                min = mid + 1;
            } else {
                max = mid;
            }
        }
        return min;
    }
    private int count(int[][] matrix, int n, int mid) {
        int i = 0, j = n - 1, count = 0;
        while (i <= n - 1 && j >= 0) {
            if (mid >= matrix[i][j]) {
                // current column;
                count += j + 1;
                i++;
            } else {
                j--;
            }
        }
        return count;
    }
    /**
     * Guess Number Higher or Lower: We are playing the Guess Game. The game is as follows:

     I pick a number from 1 to n. You have to guess which number I picked.

     Every time you guess wrong, I'll tell you whether the number is higher or lower.

     You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):

     -1 : My number is lower
     1 : My number is higher
     0 : Congrats! You got it!
     * @param args
     */
//    public int guessNumber(int n) {
//        int low = 1, high = n;
//        while (low + 1 < high) {
//            int mid = low + (high - low) / 2;
//            if (guess(mid) == -1) {
//                high = mid;
//            } else if (guess(mid) == 1) {
//                low = mid;
//            } else {
//                return mid;
//            }
//        }
//        if (guess(low) == 0) {
//            return low;
//        } else {
//            return high;
//        }
//    }

    /**
     * Longest Increasing Subsequence: Given an unsorted array of integers, find the length of longest increasing subsequence.

     For example,
     Given [10, 9, 2, 5, 3, 7, 101, 18],
     The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.

     Your algorithm should run in O(n2) complexity.

     Follow up: Could you improve it to O(n log n) time complexity? Refer to Geekforgeeks;


     * @param args
     */

    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] tailTable = new int[nums.length];
        tailTable[0] = nums[0];
        int len = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > tailTable[len - 1]) {
                tailTable[len++] = nums[i];
            } else if (nums[i] < tailTable[0]) {
                tailTable[0] = nums[i];
            } else {
                int index = findIndex(tailTable, 0, len - 1, nums[i]);
                tailTable[index] = nums[i];
            }
        }
        return len;
    }
    private int findIndex(int[] table, int start, int end, int target) {
        int index;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (table[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (table[start] >= target) {
            index = start;
        } else {
            index = end;
        }
        return index;
    }

    /**
     * H-Index II: Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize
     * your algorithm?
     * @param args
     */

    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0) {
            return 0;
        }
        int start = 0, end = citations.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (citations[mid] == citations.length - mid) {
                return citations[mid];
            } else if (citations[mid] < citations.length - mid) {
                start = mid;
            } else {
                end = mid;
            }
        }
        if (citations[start] >= citations.length - start) {
            return citations.length - start;
        } else if (citations[end] >= citations.length - end) {
            return citations.length - end ;
        } else {
            return citations.length - end - 1;
        }
    }

    /**
     * Valid Perfect Square: Given a positive integer num, write a function which returns True if num is a perfect
     * square else False.

     Note: Do not use any built-in library function such as sqrt.
     * @param args
     */

    public boolean isPerfectSquare(int num) {
        if (num == 0) {
            return true;
        }
        long start = 0, end = num;
        while (start + 1 < end) {
            long mid = start + (end - start) / 2;
            if (mid * mid == (long) num) {
                return true;
            } else if (mid * mid > (long) num) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (start * start == (long) num) {
            return true;
        }
        if (end * end == (long) num) {
            return true;
        }
        return false;
    }

    /**
     * Smallest Rectangle Enclosing Black Pixels:
     * An image is represented by a binary matrix with 0 as a white pixel and 1 as a black pixel. The black pixels are
     * connected, i.e., there is only one black region. Pixels are connected horizontally and vertically. Given the
     * location (x, y) of one of the black pixels, return the area of the smallest (axis-aligned) rectangle that
     * encloses all black pixels.

     For example, given the following image:

     [
     "0010",
     "0110",
     "0100"
     ]
     and x = 0, y = 2,
     Return 6.
     * @param image
     * @param x
     * @param y
     * @return
     */
     // Time Complexity: O(nlogm + mlogn)
    public int minArea(char[][] image, int x, int y) {
        if (image == null || image.length == 0 || image[0].length == 0) {
            return 0;
        }
        int m = image.length, n = image[0].length;
        int left = binarySearchCol(image, 0, y, 0, m, true);
        int right = binarySearchCol(image, y + 1, n, 0, m, false);
        int up = binarySearchRow(image, 0, x, 0, n, true);
        int down = binarySearchRow(image, x + 1, m, 0, n, false);
        return (right - left) * (down - up);
    }
    private int binarySearchCol(char[][] image, int start, int end, int top, int bottom, boolean touchHigh) {
        while (start != end) {
            int mid = (start + end) / 2;
            int row = top;
            while (row < bottom && image[row][mid] == '0') {
                row++;
            }
            // to the left side, if search row touch bottom, means < mid there is no 1's
            // to the right side, if search row touch bottoms, meand > mid there is no 1's
            // same for up and down search.
            if ((row == bottom) == touchHigh) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        return start;
    }
    private int binarySearchRow(char[][] image, int start, int end, int left, int right, boolean touchHigh) {
        while (start != end) {
            int mid = (start + end) / 2;
            int col = left;
            while (col < right && image[mid][col] == '0') {
                col++;
            }
            // to the left side, if search row touch bottom, means < mid there is no 1's
            // to the right side, if search row touch bottoms, meand > mid there is no 1's
            // same for up and down search.
            if ((col == right) == touchHigh) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        return start;
    }

    /**
     * Is SubSequence Follow Up:
     * Given a string s and a string t, check if s is subsequence of t.

     You may assume that there is only lower case English letters in both s and t. t is potentially a very long
     (length ~= 500,000) string, and s is a short string (<=100).

     A subsequence of a string is a new string which is formed from the original string by deleting some (can be none)
     of the characters without disturbing the relative positions of the remaining characters. (ie, "ace" is a
     subsequence of "abcde" while "aec" is not).

     Example 1:
     s = "abc", t = "ahbgdc"

     Return true.

     Example 2:
     s = "axc", t = "ahbgdc"

     Return false.

     Follow up:
     If there are lots of incoming S, say S1, S2, ... , Sk where k >= 1B, and you want to check one by one to see if T has its subsequence. In this scenario, how would you change your code?
     * @param args
     */
    // public boolean isSubsequence(String s, String t) {
    //     if (s == null || s.length() == 0) {
    //         return true;
    //     }
    //     int i = 0, j = 0;
    //     while (i < s.length() && j < t.length()) {
    //         if (t.charAt(j) == s.charAt(i)) {
    //             i++;
    //             j++;
    //         } else {
    //             j++;
    //         }
    //     }
    //     if (i == s.length()) {
    //         return true;
    //     }
    //     return false;
    // }
    // if there are lots of incoming S, we preprocess t and save it to a map <char, indexList>, every time when a new s comes, just process s char one by one and binary search the index of the char, index of char needs to be big than the previous index;
    public boolean isSubsequence(String s, String t) {
        if (s == null || s.length() == 0) {
            return true;
        }
        HashMap<Character, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            if (!map.containsKey(t.charAt(i))) {
                map.put(t.charAt(i), new ArrayList<Integer>());
            }
            map.get(t.charAt(i)).add(i);
        }
        int index = -1;
        for (int i = 0; i < s.length(); i++) {
            int nextIndex = getNextIndex(map.get(s.charAt(i)), index);
            if (nextIndex == -1) {
                return false;
            } else {
                index = nextIndex;
            }
        }
        return true;
    }
    private int getNextIndex(ArrayList<Integer> list, int index) {
        if (list == null || list.size() == 0) {
            return -1;
        }
        int start = 0, end = list.size() - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (list.get(mid) <= index) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        if (list.get(start) <= index) {
            return -1;
        } else {
            return list.get(start);
        }
    }
     public static void main(String args[]) {
        BinarySearch bs = new BinarySearch();
        int[] num1 = {1,2,3,4,6,8,80,82};
        int[] num2 = {4,5,8,9,12,15,45,60};
        double median = bs.findMedianSortedArrays(num1, num2);
        System.out.println("median of the two above arrays is: ");
        System.out.println(median);
        double power = bs.myPow(0.5, 3);
        System.out.println(0.5 + "pow of " + 3 + "is:" );
        System.out.println(power);
    }
}

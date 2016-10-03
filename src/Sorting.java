/**
 * Created by Tingting on 10/1/16.
 * MergeSort and QuickSort are written below
 * BucketSort C++ is given:
 *
 * void bucketSort(float arr[], int n)
 {
 // 1) Create n empty buckets
 vector<float> b[n];

 // 2) Put array elements in different buckets
 for (int i=0; i<n; i++)
 {
 int bi = n*arr[i]; // Index in bucket
 b[bi].push_back(arr[i]);
 }

 // 3) Sort individual buckets
 for (int i=0; i<n; i++)
 sort(b[i].begin(), b[i].end());

 // 4) Concatenate all buckets into arr[]
 int index = 0;
 for (int i = 0; i < n; i++)
 for (int j = 0; j < b[i].size(); j++)
 arr[index++] = b[i][j];
 }
 */



/**
Radix sort for positive number is given, for negative number, simply sort the sign and then reverse the negative number
array.
 */

/**
 * QuickSort: Best: O(nlogn), Average: O(nlogn), Worst: n^2, in place, not stable
 * Merge Sort: Best: O(nlogn), Average: O(nlogn), Worst: O(nlogn), Memory O(n), stable
 * Radix Sort: When h = l * lg n for some constant l > 0, the range of the input
 elements is {0, 1, . . . , n
 n ^ l − 1}. If we set r = lg n, then the total
 running time of Radix sort is
 O(l(n + 2r))= O(l · n) = O(n)
 */


import java.util.Arrays;
public class Sorting {
    public void mergeSort(int[] array) {
        int start = 0, end = array.length - 1;
        mergeSortHelper(array, start, end);
    }
    private void mergeSortHelper(int[] array, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = (start + end) / 2;
        mergeSortHelper(array, start, mid);
        mergeSortHelper(array, mid + 1, end);
        merge(array, start, mid, end);
    }
    private void merge(int[] array, int start, int mid, int end) {
        if (start >= end) {
            return;
        }
        int l = mid - start + 1, r = end - mid;
        int[] L = new int[l], R = new int[r];
        for (int i = 0; i < l; i++) {
            L[i] = array[start + i];
        }
        for (int i = 0; i < r; i++) {
            R[i] = array[mid + 1 + i];
        }
        int i = 0, j = 0, k = start;
        while (i < l && j < r) {
            if (L[i] < R[j]) {
                array[k] = L[i];
                i++;
            } else {
                array[k] = R[j];
                j++;
            }
            k++;
        }
        while (i < l) {
            array[k] = L[i];
            i++;
            k++;
        }
        while (j < r) {
            array[k] = R[j];
            j++;
            k++;
        }
    }

    /**
     * MergeSort LinkedList
     * @param
     */
    public SListNode mergeSortLinkedList(SListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        SListNode middle = findMiddle(head);
        SListNode head1 = mergeSortLinkedList(middle.next);
        middle.next = null;
        SListNode head2 = mergeSortLinkedList(head);
        head = merge(head1, head2);
        return head;
    }
    private SListNode findMiddle(SListNode head) {
        SListNode slow = head;
        SListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    private SListNode merge(SListNode head1, SListNode head2) {
        SListNode dummy = new SListNode(0);
        SListNode current = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                current.next = head1;
                head1 = head1.next;
            } else {
                current.next = head2;
                head2 = head2.next;
            }
            current = current.next;
        }
        while (head1 != null) {
            current.next = head1;
            head1 = head1.next;
            current = current.next;
        }
        while (head2 != null) {
            current.next = head2;
            head2 = head2.next;
            current = current.next;
        }
        return dummy.next;
    }

    public void quickSort(int[] array) {
        quickSortHelper(array, 0, array.length - 1);
    }
    private void quickSortHelper(int[] array, int start, int end) {
        if (start >= end) {
            return;
        }
        int left = start, right = end, pivot = array[(start + end) / 2];
        while (left <= right) {
            while (left <= right && array[left] < pivot) {
                left++;
            }
            while (left <= right && array[right] > pivot) {
                right--;
            }
            if (left <= right) {
                swap(array, left, right);
                left++;
                right--;
            }
        }
        quickSortHelper(array, start, right);
        quickSortHelper(array, left, end);
    }
    private void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }


    /**
     * QuickSort LinkedList
     * @param
     */

    public SListNode quickSortList(SListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        SListNode middle = findMiddle(head);
        SListNode leftDummy = new SListNode(0), leftTail = leftDummy;
        SListNode rightDummy = new SListNode(0), rightTail = rightDummy;
        SListNode middleDummy = new SListNode(0), middleTail = middleDummy;
        while (head != null) {
//            System.out.println(head.val);
            if (head.val < middle.val) {
                leftTail.next = head;
                leftTail = head;
//                System.out.println(leftTail.val);
            } else if (head.val > middle.val) {
                rightTail.next = head;
                rightTail = head;
//                System.out.println(rightTail.val);
            } else {
                middleTail.next = head;
                middleTail = head;
//                System.out.println(middleTail.val);
            }
            head = head.next;
        }
//        System.out.println(leftDummy.next.val);
//        System.out.println(leftTail.val);
        leftTail.next = null;
        rightTail.next = null;
        middleTail.next = null;
        SListNode left = quickSortList(leftDummy.next);
        SListNode right = quickSortList(rightDummy.next);
        return concat(left, middleDummy.next, right);
    }
    private SListNode concat(SListNode left, SListNode middle, SListNode right) {
        SListNode dummy = new SListNode(0), tail = dummy;
        while (left != null) {
//            System.out.println(left.val);
            tail.next = left;
            left = left.next;
            tail = tail.next;
        }
        while (middle != null) {
//            System.out.print(middle.val);
            tail.next = middle;
            middle = middle.next;
            tail = tail.next;
        }
        while (right != null) {
            tail.next = right;
            right = right.next;
            tail = tail.next;
        }
        return dummy.next;
    }


    public void radixSort(int[] array) {
        int n = array.length;
        int m = getMax(array);
        for (int exp = 1; m / exp > 0; exp *= 10) {
            countSort(array, n, exp);
        }
    }
    private int getMax(int[] array) {
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            max = Math.max(max, array[i]);
        }
        return max;
    }
    private void countSort(int[] array, int size, int exp) {
        int[] output = new int[size];
        int[] count = new int[10];
        Arrays.fill(count, 0);
        for (int i = 0; i < size; i++) {
            count[(array[i] / exp) % 10]++;
        }
        // Change count[i] so that count[i] now contains
        // actual position of this digit in output[]
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        for (int i = size -1; i >= 0; i--) {
            // count[... % 10] is the number of these array item less than and equal to ...%10, -1 is to find the index
            output[count[(array[i] / exp) % 10] - 1] = array[i];
            count[(array[i] / exp) % 10]--;
        }
        for (int i = 0; i < size; i++) {
            array[i] = output[i];
        }
    }
//    public static void main(String args[]) {
//        Sorting S = new Sorting();
//        int[] array = {4,1,2,8,11,8,-1,3,6,-20, 3, 5, 11, 20,3,45,27,23,-3,-5};
//        int[] array2 = {111, 20, 13, 24, 333, 45, 56, 37, 18, 38, 235, 446, 98, 89, 100};
//        S.quickSort(array);
//        S.radixSort(array2);
////        S.mergeSort(array);
////        for (int i = 0; i < array2.length; i++) {
////            System.out.print(array2[i]);
////            System.out.print(",");
////        }
//        SListNode node1 = new SListNode(4);
//        SListNode node2 = new SListNode(1);
//        SListNode node3 = new SListNode(2);
//        SListNode node4 = new SListNode(8);
//        SListNode node5 = new SListNode(11);
//        SListNode node6 = new SListNode(8);
//        SListNode node7 = new SListNode(-1);
//        node1.next = node2;
//        node2.next = node3;
//        node3.next = node4;
//        node4.next = node5;
//        node5.next = node6;
//        node6.next = node7;
////        SListNode sorted = S.mergeSortLinkedList(node1);
//        SListNode quickSorted = S.quickSortList(node1);
//        while (quickSorted != null) {
//            System.out.println(quickSorted.val);
//            quickSorted = quickSorted.next;
//        }
//    }
}

import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by Tingting on 10/2/16.
 */

/**
 * Design an iterator over a binary search tree with the following rules:

 Elements are visited in ascending order (i.e. an in-order traversal)
 next() and hasNext() queries run in O(1) time in average.
 */
class BSTIterator{
    public Stack<TreeNode> stack;
    public BSTIterator(TreeNode root) {
        stack = new Stack<TreeNode>();
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
    }
    //@return: True if there has next node, or false
    public boolean hasNext() {
        // write your code here
        return !stack.isEmpty();
    }

    //@return: return next node
    public TreeNode next() {
        // write your code here
        TreeNode node = stack.pop();
        TreeNode result = node;
        if (node.right != null) {
            node = node.right;
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
        }
        return result;
    }
}
public class Tree {
    /**
     * minimum and maximum depth of binary tree
     * @param args
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return minDepth(root.right) + 1;
        }
        if (root.right == null) {
            return minDepth(root.left) + 1;
        }
        return Math.min(minDepth(root.right), minDepth(root.left)) + 1;
    }
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return maxDepth(root.right) + 1;
        }
        if (root.right == null) {
            return maxDepth(root.left) + 1;
        }
        return Math.max(maxDepth(root.right), maxDepth(root.left)) + 1;
    }

    /**
     * Tree traversal, preorder, inorder and postorder and levelorder, both recursive and non-recursive
     * @param args
     */

    public ArrayList<Integer> preorderTraversalRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        preorderTraverse(root, result);
        return result;
    }
    private void preorderTraverse(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        result.add(root.val);
        preorderTraverse(root.left, result);
        preorderTraverse(root.right, result);
    }
    public ArrayList<Integer> preorderTraversalNoRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return result;
    }



    public ArrayList<Integer> inorderTraversalRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        inorderTraverse(root, result);
        return result;
    }
    private void inorderTraverse(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        inorderTraverse(root.left, result);
        result.add(root.val);
        inorderTraverse(root.right, result);
    }

    public ArrayList<Integer> inorderTraversalNoRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            result.add(root.val);
            root = root.right;
        }
        return result;
    }

    public ArrayList<Integer> postorderTraversalRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        postorderTraverse(root, result);
        return result;
    }
    private void postorderTraverse(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        postorderTraverse(root.left, result);
        postorderTraverse(root.right, result);
        result.add(root.val);
    }


    public ArrayList<Integer> postorderTraversalNoRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode prev = null, curr= root;
        while (!stack.isEmpty()) {
            curr = stack.peek();
            // from top to bottom, pre is above cur
            if (prev == null || prev.left == curr || prev.right == curr) {
                if (curr.left != null) {
                    stack.push(curr.left);
                } else if (curr.right != null) {
                    stack.push(curr.right);
                } else {
                    result.add(curr.val);
                    stack.pop();
                }
                // backward, pre is below cur
            } else if (curr.left == prev && curr.right != null) {
                stack.push(curr.right);
                // no right child, pop out;
            } else {
                result.add(curr.val);
                stack.pop();
            }
            prev = curr;
        }
        return result;
    }

    public ArrayList<Integer> levelorderTraversalNoRecur(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            result.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }
        return result;
    }

    /**
     * Zigzag level Order Traversal
     */
    public ArrayList<ArrayList<Integer>> zigzagTraversal(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int size = queue.size();
        int count = 0;
        boolean flag = true;
        ArrayList<Integer> list = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (flag) {
                list.add(node.val);
            } else {
                list.add(0, node.val);
            }
            count++;
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
            if (count == size) {
                size = queue.size();
                result.add(list);
                list = new ArrayList<Integer>();
                count = 0;
                flag = !flag;
            }
        }
        return result;
    }

    /**
     * Check if two trees are identical, or tweaked Identical
     */

    public boolean isIdentical(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null) {
            return false;
        }
        if (node1.val == node2.val) {
            return isIdentical(node1.left, node2.left) && isIdentical(node1.right, node2.right);
        }
        return false;
    }

    /**
     * Check two given binary trees are identical or not. Assuming any number of tweaks are allowed.
     * A tweak is defined as a swap of the children of one node in the tree.
     * @param node1
     * @param node2
     * @return
     */
    public boolean isTweakedIdentical(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null) {
            return false;
        }
        if (node1.val == node2.val) {
            return ((isTweakedIdentical(node1.right, node2.right) && isTweakedIdentical(node1.left, node2.left)) || (isTweakedIdentical(node1.left, node2.right) && isTweakedIdentical(node1.right, node2.left)));
        }
        return false;
    }

    /**
     * Symmetric Binary Tree: check if a tree is symmetric around its center
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return checkSymmetric(root.left, root.right);
    }
    private boolean checkSymmetric(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null) {
            return false;
        }
        return node1.val == node2.val && (checkSymmetric(node1.left, node2.right) && checkSymmetric(node1.right, node2.left));
    }

    /**
     * Is Complete: Check a binary tree is completed or not. A complete binary tree is a binary tree that every level
     * is completed filled except the deepest level. In the deepest level, all nodes must be as left as possible.
     */
    // Recursion Method: define a result type with depth, isFull, isComplete, if leftdepth == rightdepth, left needs
    // to be full, right needs to be complete, if leftdepth = rightdepth + 1, right needs to be full, left needs to be
    // complete
    // Here we give non recursion method: using a queue
    public boolean isComplete(TreeNode root) {
        if (root == null) {
            return true;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        int index = 0;
        queue.add(root);
        while (index < queue.size()) {
            TreeNode node = queue.get(index);
            index++;
            if (node == null) {
                continue;
            } else {
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }
        while (queue.get(queue.size() - 1) == null) {
            queue.remove(queue.size() - 1);
        }
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                return false;
            }
        }
        return true;
    }

    /**
     * Is balanced: Given a binary tree, determine if it is height-balanced.

     For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees
     of every node never differ by more than 1.
     */
    public boolean isBalenced(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (getHeight(root) == -1) {
            return false;
        }
        return true;
    }
    private int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = getHeight(root.left);
        int right = getHeight(root.right);
        if (left == -1 || right == -1) {
            return -1;
        }
        if (Math.abs(left - right) > 1) {
            return -1;
        }
        return Math.max(left, right) + 1;
    }

    /**
     * Valid Binary Search Tree:
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isValidBSThelper(root, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }
    private boolean isValidBSThelper(TreeNode root, Double min, Double max) {
        if (root == null) {
            return true;
        }
        if (root.val >= max || root.val <= min) {
            return false;
        }
        boolean isLeftValid = isValidBSThelper(root.left, min, (double) root.val);
        boolean isRightValid = isValidBSThelper(root.right, (double) root.val, max);
        return isLeftValid && isRightValid;
    }
    /**
     * Inorder Successor in Binary Search Tree
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode A) {
        TreeNode successor = null;
        while (root != null && root != A) {
            // record the last node that has value greater than A
            if (root.val > A.val) {
                successor = root;
                root = root.left;
            } else {
                root = root.right;
            }
        }
        if (root == null) {
            return null;
        }
        if (root.right == null) {
            return successor;
        }
        // find right child's most left child
        root = root.right;
        while (root.left != null) {
            root = root.left;
        }
        return root;
    }
    /**
     * Lowest Common Ancestor I, II, III, LCA with parent reference, use a hashset to record A and A's all parents
     * then for B do the same thing to check if B's parent is already in the hashset, if yes return, no, return null
     *
     */
    public TreeNode LCA(TreeNode root, TreeNode A, TreeNode B) {
        if (root == null || root == A || root == B) {
            return root;
        }
        TreeNode left = LCA(root.left, A, B);
        TreeNode right = LCA(root.right, A, B);
        if (left != null && right != null) {
            return root;
        }
        if (left != null) {
            return left;
        }
        if (right != null) {
            return right;
        }
        return null;
    }


    /**
     * Invert Binary Tree
     * @param args
     */
    public void invertBinaryTree(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode temp = root.right;
        root.right = root.left;
        root.left = temp;
        invertBinaryTree(root.left);
        invertBinaryTree(root.right);
    }


    /**
     * Path Sum 类问题
     * @param args
     */


    /**
     * Binary Tree Maximum Path Sum: Given a binary tree, find the maximum path sum.

     The path may start and end at any node in the tree.
     * @param args
     */

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int[] sum = new int[1];
        sum[0] = Integer.MIN_VALUE;
        //singlePath is the maximal sum from root to one side
        int singlePath = maxPathSumHelper(root, sum);
        return sum[0];
    }
    private int maxPathSumHelper(TreeNode root, int[] sum) {
        if (root == null) {
            return 0;
        }
        int leftSinglePath = Math.max(maxPathSumHelper(root.left, sum), 0);
        int rightSinglePath = Math.max(maxPathSumHelper(root.right, sum), 0);
        int singlePath = Math.max(leftSinglePath, rightSinglePath) + root.val;
        int twoPath = leftSinglePath + root.val + rightSinglePath;
        sum[0] = Math.max(sum[0], Math.max(singlePath, twoPath));
        return singlePath;
    }

    /**
     * maxPathSum II: Given a binary tree, find the maximum path sum from root.

     The path may end at any node in the tree and contain at least one node in it.
     */
    public int maxPathSumII(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = Math.max(maxPathSumII(root.left), 0);
        int right = Math.max(maxPathSumII(root.right), 0);
        return Math.max(left, right) + root.val;
    }

    /**
     * Path Sum: Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all
     * the values along the path equals the given sum.
     * @param args
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (sum == root.val && root.left == null && root.right == null) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    /**
     * Path Sum II: Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
     */

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> results = new ArrayList<>();
        if (root == null) {
            return results;
        }
        List<Integer> list = new ArrayList<>();
        list.add(root.val);
        pathSumHelper(root, sum - root.val, results, list);
        return results;
    }
    private void pathSumHelper(TreeNode root, int sum, List<List<Integer>> results, List<Integer> list) {
        if (root.left == null && root.right == null && sum == 0) {
            results.add(new ArrayList<Integer>(list));
        }
        if (root.left != null) {
            list.add(root.left.val);
            pathSumHelper(root.left, sum - root.left.val, results, list);
            list.remove(list.size() - 1);
        }
        if (root.right != null) {
            list.add(root.right.val);
            pathSumHelper(root.right, sum - root.right.val, results, list);
            list.remove(list.size() - 1);
        }
    }

    /**
     * Tree Reconstruction Problem: From sorted array, from pre and in order traversal, from post and in order traversal
     * Tree Serialization and deserialization
     * @param args
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }
    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

    public TreeNode buildTreePostAndIn(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null || inorder.length == 0 || postorder.length == 0) {
            return null;
        }
        int m = inorder.length;
        return buildTreePostAndIn(inorder, 0, m - 1, postorder, 0, m - 1);
    }
    private TreeNode buildTreePostAndIn(int[] inorder, int istart, int iend, int[] postorder, int pstart, int pend) {
        if (istart > iend || pstart > pend) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[pend]);
        int k = 0;
        for (int j = 0; j < inorder.length; j++) {
            if (inorder[j] == root.val) {
                k = j;
                break;
            }
        }
        // left tree length is k - istart
        root.left = buildTreePostAndIn(inorder, istart, k - 1, postorder, pstart, pstart + k - istart - 1);
        root.right = buildTreePostAndIn(inorder, k + 1, iend, postorder, pstart + k - istart, pend - 1);
        return root;
    }

    public TreeNode buildTreePreAndIn(int[] inorder, int[] preorder) {
        if (inorder == null || preorder == null || inorder.length == 0 || preorder.length == 0) {
            return null;
        }
        int m = inorder.length;
        return buildTreePreAndIn(inorder, 0, m - 1, preorder, 0, m - 1);
    }
    private TreeNode buildTreePreAndIn(int[] inorder, int istart, int iend, int[] preorder, int pstart, int pend) {
        if (istart > iend || pstart > pend) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[pstart]);
        int k = 0;
        for (int j = 0; j < inorder.length; j++) {
            if (inorder[j] == root.val) {
                k = j;
                break;
            }
        }
        root.left = buildTreePreAndIn(inorder, istart, k - 1, preorder, pstart + 1, pstart + 1 + k - istart - 1);
        root.right = buildTreePreAndIn(inorder, k + 1, iend, preorder, pstart + k - istart + 1, pend);
        return root;
    }

    /**
     * Binary Tree Serialization: Design an algorithm and write code to serialize and deserialize a binary tree. Writing
     * the tree to a file is called 'serialization' and reading back from the file to reconstruct the exact same binary
     * tree is 'deserialization'.

     There is no limit of how you deserialize or serialize a binary tree, you only need to make sure you can serialize a
     binary tree to a string and deserialize this string to the original structure.
     * @param args
     */
    public String serialize(TreeNode root) {
        if (root == null) {
            return "";
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int index = 0;
        while (index < queue.size()) {
            TreeNode node = queue.get(index);
            index++;
            if (node == null) {
                continue;
            }
            queue.add(node.left);
            queue.add(node.right);
        }
        while (queue.get(queue.size() - 1) == null) {
            queue.remove(queue.size() - 1);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < queue.size(); i++) {
            TreeNode node = queue.get(i);
            if (node == null) {
                sb.append("#,");
            } else {
                sb.append(String.valueOf(node.val));
                sb.append(",");
            }
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    public TreeNode deserialize(String s) {
        if (s == "") {
            return null;
        }
        boolean isLeft = true;
        String[] tree = s.split(",");
        TreeNode root = new TreeNode(Integer.parseInt(tree[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int index = 1;
        while (index < tree.length) {
            TreeNode node = queue.poll();
            if (node == null) {
                index++;
                continue;
            }
            if (isLeft) {
                if (tree[index].equals("#")) {
                    node.left = null;
                } else {
                    node.left = new TreeNode(Integer.parseInt(tree[index]));
                }
                isLeft = !isLeft;
                index++;
                queue.add(node.left);
            }
            if (index < tree.length) {
                if (!isLeft) {
                    if (tree[index].equals("#")) {
                        node.right = null;
                    } else {
                        node.right = new TreeNode(Integer.parseInt(tree[index]));
                    }
                    isLeft = !isLeft;
                    index++;
                    queue.add(node.right);
                }
            }
        }
        return root;
    }

    /**
     * Insert and remove node in a binary search tree
     * @param args
     */
    public TreeNode insertNode(TreeNode root, TreeNode node) {
        if (root == null) {
            return node;
        }
        TreeNode prev = null, curr = root;
        while (curr != null) {
            if (node.val > curr.val) {
                prev = curr;
                curr = curr.right;
            } else {
                prev = curr;
                curr = curr.left;
            }
        }
        if (prev.val > node.val) {
            prev.left = node;
        } else {
            prev.right = node;
        }
        return root;
    }

    public TreeNode removeNode(TreeNode root, TreeNode node) {
        if (root == null) {
            return root;
        }
        if (root.val < node.val) {
            root.right = removeNode(root.right, node);
        } else if (root.val > node.val) {
            root.left = removeNode(root.left, node);
        } else {
            if (root.left == null) {
                return root.right;
            }
            if (root.right == null) {
                return root.left;
            }
            TreeNode temp = root;
            root = findMin(root.right);
            root.right = removeNode(temp.right, root);
            root.left = temp.left;
        }
        return root;
    }
    private TreeNode findMin(TreeNode root) {
        while (root.left != null) {
            root = root.left;
        }
        return root;
    }

    /**
     * Search Range in Binary Search Tree: find k1 <= x <= k2
     * @param args
     */

    public ArrayList<Integer> searchRange(TreeNode root, int k1, int k2) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null || k1 > k2) {
            return result;
        }
        searchRangeHelper(root, k1, k2, result);
        return result;
    }
    private void searchRangeHelper(TreeNode root, int k1, int k2, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        if (root.val >= k1 && root.val <= k2) {
            result.add(root.val);
            searchRangeHelper(root.left, k1, k2, result);
            searchRangeHelper(root.right, k1, k2, result);
        } else if (root.val < k1) {
            searchRangeHelper(root.right, k1, k2, result);
        } else {
            searchRangeHelper(root.left, k1, k2, result);
        }
    }

    /**
     * Flatten Binary Tree to Linked List
     * @param args
     */
    public void flatten(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return;
        }
        root = flattenHelper(root);
    }
    private TreeNode flattenHelper(TreeNode root) {
        TreeNode node = root;
        if (node == null) {
            return null;
        }
        TreeNode temp = node.right;
        node.right = flattenHelper(node.left);
        node.left = null;
        while (node.right != null) {
            node = node.right;
        }
        node.right = flattenHelper(temp);
        return root;
    }

    /**
     * Print Binary Tree Paths
     * @param args
     */

    public ArrayList<String>  binaryTreePaths(TreeNode root) {
        ArrayList<String> paths = new ArrayList<>();
        if (root == null) {
            return paths;
        }
        binaryTreePathsHelper(root, String.valueOf(root.val), paths);
        return paths;
    }
    private void binaryTreePathsHelper(TreeNode root, String path, ArrayList<String> paths) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            paths.add(path);
        }
        if (root.left != null) {
            binaryTreePathsHelper(root.left, path + "->" + String.valueOf(root.left.val), paths);
        }
        if (root.right != null) {
            binaryTreePathsHelper(root.right, path + "->" + String.valueOf(root.right.val), paths);
        }
    }

    /**
     * populating Next Right Pointers in Each Node
     * @param args
     */
    public void connect(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode leftMarker = root.left;
        TreeNode parent = root;
        while (leftMarker != null) {
            TreeNode currentMarker = leftMarker;
            while (currentMarker != null && parent != null) {
                if (currentMarker == parent.left) {
                    currentMarker.next = parent.right;
                    currentMarker = currentMarker.next;
                } else {
                    parent = parent.next;
                    if (parent != null) {
                        currentMarker.next = parent.left;
                        currentMarker = currentMarker.next;
                    }
                }
            }
            parent = leftMarker;
            leftMarker = parent.left;
        }
    }

    /**
     * Populationg Next Right Pointers in Each Node II: What if the given tree could be any binary tree? Would your
     * previous solution still work?
     */
    public void connectII(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode leftMarker;
        if (root.left != null) {
            leftMarker = root.left;
        } else {
            leftMarker = root.right;
        }
        TreeNode parent = root;
        while (leftMarker != null) {
            TreeNode currentMarker = leftMarker;
            while (currentMarker != null && parent != null) {
                if (currentMarker == parent.left) {
                    if (parent.right != null) {
                        currentMarker.next = parent.right;
                        currentMarker = currentMarker.next;
                    }
                } else {
                    parent = parent.next;
                    if (parent != null) {
                        if (parent.left != null) {
                            currentMarker.next = parent.left;
                            currentMarker = currentMarker.next;
                        } else if (parent.right != null) {
                            currentMarker.next = parent.right;
                            currentMarker = currentMarker.next;
                        }
                    }
                }
            }
            parent = leftMarker;
            while (parent.left == null &&  parent.right == null) {
                parent = parent.next;
            }
            if (parent.left != null) {
                leftMarker = root.left;
            } else if (parent.right != null) {
                leftMarker = root.right;
            }
        }
    }

    /**
     * Recover Binary Search Tree: Two elements of a binary search tree (BST) are swapped by mistake.
     Recover the tree without changing its structure.
     * @param args
     */

    private TreeNode first = null;
    private TreeNode second = null;
    private TreeNode prev = new TreeNode(Integer.MIN_VALUE);
    public void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        traverse(root.left);
        if (first == null && root.val < prev.val) {
            first = prev;
        }
        if (first != null && prev.val > root.val) {
            second = root;
        }
        prev = root;
        traverse(root.right);
    }
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        traverse(root);
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }


    /**
     * Count Complete Tree Nodes:In a complete binary tree every level, except possibly the last, is completely filled,
     * and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the
     * last level h.
     *
     * First check if leftmostheight == rightmostheight, if yes, 2^h - 1, if no, count left + count right + 1
     * @param args
     */

    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftMostHeight = getLeftHeight(root);
        int rightMostHeight = getRightHeight(root);
        if (leftMostHeight == rightMostHeight) {
            return (1 << leftMostHeight) - 1;
        } else {
            return countNodes(root.left) + countNodes(root.right) + 1;
        }
    }
    private int getLeftHeight(TreeNode root) {
        int count = 0;
        while (root != null) {
            count++;
            root = root.left;
        }
        return count;
    }
    private int getRightHeight(TreeNode root) {
        int count = 0;
        while (root != null) {
            count++;
            root = root.right;
        }
        return count;
    }

    /**
     * Unique Binary Search Trees II: Given an integer n, generate all structurally unique BST's (binary search trees)
     * that store values 1...n.
     * @param args
     */

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new ArrayList<TreeNode>();
        return generateTreesHelper(1, n);
    }
    private List<TreeNode> generateTreesHelper(int start, int end) {
        List<TreeNode> result = new ArrayList<TreeNode>();
        if (start > end) {
            // lastnode, left and right are null
            result.add(null);
            return result;
        }
        for (int i = start; i <= end; i++) {
            // all left tree
            List<TreeNode> left = generateTreesHelper(start, i - 1);
            // all right tree
            List<TreeNode> right = generateTreesHelper(i + 1, end);
            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode cur = new TreeNode(i);
                    cur.left = l;
                    cur.right = r;
                    result.add(cur);
                }
            }
        }
        return result;
    }

    /**
     * Kth smallest Element in BST
     * @param args
     */
    public int kthSmallest(TreeNode root, int k) {
        if (root == null) {
            return -1;
        }
        Stack<TreeNode> stack = new Stack<>();
        int count = 0;
        int result = 0;
        while (root != null || !stack.isEmpty()) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                TreeNode node = stack.pop();
                count++;
                if (count == k) {
                    result = node.val;
                    break;
                }
                root = node.right;
            }
        }
        return result;
    }

    /**
     * Binary Tree Right Side View: Given a binary tree, imagine yourself standing on the right side of it, return the
     * values of the nodes you can see ordered from top to bottom.
     * @param root
     * @return
     */

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int size = queue.size();
        TreeNode node = null;
        while (!queue.isEmpty()) {
            for (int i = size; i > 0; i--) {
                node = queue.poll();
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            result.add(node.val);
            size = queue.size();
        }
        return result;
    }

    /**
     * Sum Root to Leaf Numbers: Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

     An example is the root-to-leaf path 1->2->3 which represents the number 123.

     Find the total sum of all root-to-leaf numbers.
     * @param root
     * @return
     */

    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int prev = 0;
        return sumNumbersHelper(root, prev);
    }
    private int sumNumbersHelper(TreeNode root, int prev) {
        if (root == null) {
            return 0;
        }
        int sum = prev * 10 + root.val;
        if (root.left == null && root.right == null) {
            return sum;
        } else {
            return sumNumbersHelper(root.left, sum) + sumNumbersHelper(root.right, sum);
        }
    }

    /**
     * Binary Tree Upside Down: Given a binary tree where all the right nodes are either leaf nodes with a sibling
     * (a left node that shares the same parent node) or empty, flip it upside down and turn it into a tree where the
     * original right nodes turned into left leaf nodes. Return the new root.
     * Given a binary tree {1,2,3,4,5},return the root of the binary tree [4,5,2,#,#,3,1].
     * @param args
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        // newroot is the upsideDown of the old root's left child
        TreeNode newroot = upsideDownBinaryTree(root.left);
        // find new root's right most node, and put root.right to the right most node.left
        TreeNode rightMostNode = newroot;
        while (rightMostNode.right != null) {
            rightMostNode = rightMostNode.right;
        }
        rightMostNode.left = root.right;
        // since root is still connected to root.right and left, so make a node with root.val.
        // Instead of doing this, you can set up root.right = null && root.left = null and
        // write rightMostNode.right = root;
        rightMostNode.right = new TreeNode(root.val);
        return newroot;
    }

    /**
     * Binary Tree Longest Consecutive Sequence: Given a binary tree, find the length of the longest consecutive
     * sequence path.

     The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child
     connections. The longest consecutive path need to be from parent to child (cannot be the reverse).


     * @param args
     */
    public int longestConsecutive(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> sizeQueue = new LinkedList<>();
        nodeQueue.offer(root);
        sizeQueue.offer(1);
        int max = Integer.MIN_VALUE;
        while (!nodeQueue.isEmpty()) {
            TreeNode node = nodeQueue.poll();
            int size = sizeQueue.poll();
            if (node.left != null) {
                nodeQueue.offer(node.left);
                if (node.left.val - 1 == node.val) {
                    sizeQueue.offer(size + 1);
                } else {
                    sizeQueue.offer(1);
                }
            }
            if (node.right != null) {
                nodeQueue.offer(node.right);
                if (node.right.val - 1 == node.val) {
                    sizeQueue.offer(size + 1);
                } else {
                    sizeQueue.offer(1);
                }
            }
            max = Math.max(max, size);
        }
        return max;
    }

    /**
     * Closest Binary Search Tree Value: Given a non-empty binary search tree and a target value, find the value in
     * the BST that is closest to the target.

     Note:
     Given target value is a floating point.
     You are guaranteed to have only one unique value in the BST that is closest to the target.
     Hide Company Tags
     * @param args
     */
    public int closestValue(TreeNode root, double target) {
        if (root == null) {
            return 0;
        }
        int closest = root.val;
        while (root != null) {
            closest = Math.abs(closest - target) < Math.abs(root.val - target) ? closest : root.val;
            root = root.val < target ? root.right : root.left;
        }
        return closest;
    }

    /**
     * Closest Binary Search Tree Value II: Given a non-empty binary search tree and a target value, find k values in
     * the BST that are closest to the target.
     * @param args
     */
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> predecessor = new Stack<>();
        getPredecessor(root, target, predecessor);
        Stack<TreeNode> successor = new Stack<>();
        getSuccessor(root, target, successor);
        for (int i = 0; i < k; i++) {
            if (predecessor.isEmpty() && successor.isEmpty()) {
                break;
            } else if (predecessor.isEmpty()) {
                result.add(successor.pop().val);
            } else if (successor.isEmpty()) {
                result.add(predecessor.pop().val);
            } else {
                TreeNode pre = predecessor.peek();
                TreeNode suc = successor.peek();
                if (Math.abs(pre.val - target) < Math.abs(suc.val - target)) {
                    result.add(pre.val);
                    predecessor.pop();
                } else {
                    result.add(suc.val);
                    successor.pop();
                }
            }
        }
        return result;
    }
    private void getPredecessor(TreeNode root, double target, Stack<TreeNode> stack) {
        if (root == null) {
            return;
        }
        // search from left so that it's from small to large in the stack
        getPredecessor(root.left, target, stack);
        if (root.val > target) {
            return;
        }
        stack.push(root);
        getPredecessor(root.right, target, stack);
    }
    private void getSuccessor(TreeNode root, double target, Stack<TreeNode> stack) {
        if (root == null) {
            return;
        }
        // search from right so that it's from large to small in the stack
        getSuccessor(root.right, target, stack);
        if (root.val <= target) {
            return;
        }
        stack.push(root);
        getSuccessor(root.left, target, stack);
    }

    /**
     * Count Univalue Subtrees: Given a binary tree, count the number of uni-value subtrees.

     A Uni-value subtree means all nodes of the subtree have the same value.
     * @param root
     * @return
     */

    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (isUnivalSubtree(root)) {
            return countUnivalSubtrees(root.right) + countUnivalSubtrees(root.left) + 1;
        } else {
            return countUnivalSubtrees(root.right) + countUnivalSubtrees(root.left);
        }
    }
    private boolean isUnivalSubtree(TreeNode root) {
        if (root.left == null && root.right == null) {
            return true;
        }
        if (root.left == null) {
            return root.val == root.right.val && isUnivalSubtree(root.right);
        }
        if (root.right == null) {
            return root.val == root.left.val && isUnivalSubtree(root.left);
        }
        return root.val == root.right.val && root.val == root.left.val && isUnivalSubtree(root.right) && isUnivalSubtree(root.left);
    }

    /**
     * Verify Preorder Sequence in Binary Search Tree: Given an array of numbers, verify whether it is the correct
     * preorder traversal sequence of a binary search tree.
     * For inorder, just need to verify ascending order, for postorder, started from the end of the array and update max
     * @param preorder
     * @return
     */
    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return true;
        }
        int min = Integer.MIN_VALUE;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < preorder.length; i++) {
            if (preorder[i] < min) {
                return false;
            }
            while (!stack.isEmpty() && stack.peek() < preorder[i]) {
                min = stack.pop();
            }
            stack.push(preorder[i]);
        }
        return true;
    }

    /**
     *  Find Leaves of Binary Tree: Given a binary tree, collect a tree's nodes as if you
     * were doing this: Collect and remove all leaves, repeat until the tree is empty.
     * @param root
     * @return
     */
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> leaves = new ArrayList<>();
        if (root == null) {
            return leaves;
        }
        findLeavesHelper(root, leaves);
        return leaves;
    }
    private int findLeavesHelper(TreeNode root, List<List<Integer>> list) {
        if (root == null) {
            return -1;
        }
        int left = findLeavesHelper(root.left, list);
        int right = findLeavesHelper(root.right, list);
        int curr = Math.max(left, right) + 1;
        if (list.size() <= curr) {
            list.add(new ArrayList<Integer>());
        }
        list.get(curr).add(root.val);
        return curr;
    }

    /**
     * Sum of Left Leaves:
     */


    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int sum = 0;
        if (isLeaf(root.left)) {
            sum += root.left.val;
        } else {
            sum += sumOfLeftLeaves(root.left);
        }
        sum += sumOfLeftLeaves(root.right);
        return sum;
    }
    private boolean isLeaf(TreeNode node) {
        if (node == null) {
            return false;
        } else if (node.left == null && node.right == null) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Largest BST Subtree: Given a binary tree, find the largest subtree which is a Binary Search Tree (BST),
     * where largest means subtree with largest number of nodes in it.
     */
    class Wrapper{
        int size;
        int lower, upper;
        boolean isBST;

        public Wrapper(){
            lower = Integer.MAX_VALUE;
            upper = Integer.MIN_VALUE;
            isBST = false;
            size = 0;
        }
    }
    public int largestBSTSubtree(TreeNode root) {
        return helper(root).size;
    }

    public Wrapper helper(TreeNode node){
        Wrapper curr = new Wrapper();

        if(node == null){
            curr.isBST= true;
            return curr;
        }

        Wrapper l = helper(node.left);
        Wrapper r = helper(node.right);

        //current subtree's boundaries
        curr.lower = Math.min(node.val, l.lower);
        curr.upper = Math.max(node.val, r.upper);

        //check left and right subtrees are BST or not
        //check left's upper again current's value and right's lower against current's value
        if(l.isBST && r.isBST && l.upper<=node.val && r.lower>=node.val){
            curr.size = l.size+r.size+1;
            curr.isBST = true;
        }else{
            curr.size = Math.max(l.size, r.size);
            curr.isBST  = false;
        }

        return curr;
    }

//    public static void main(String args[]) {
//        TreeNode root = new TreeNode(8);
//        TreeNode left1 = new TreeNode(3);
//        root.left = left1;
//        TreeNode right1 = new TreeNode(10);
//        root.right = right1;
//        TreeNode left2 = new TreeNode(1);
//        left1.left = left2;
//        TreeNode right2 = new TreeNode(6);
//        left1.right = right2;
//        TreeNode right3 = new TreeNode(14);
//        right1.right = right3;
//        TreeNode left4 = new TreeNode(4);
//        right2.left = left4;
//        TreeNode right4 = new TreeNode(7);
//        right2.right = right4;
//        TreeNode left5 = new TreeNode(13);
//        right3.left = left5;
//        /**
//         *                            8
//         *                          /  \
//         *                         3   10
//         *                        / \   \
//         *                       1  6   14
//         *                         / \  /
//         *                        4  7 13
//          */
//        TreeNode rootDup = new TreeNode(8);
//        TreeNode left1Dup = new TreeNode(3);
//        rootDup.left = left1Dup;
//        TreeNode right1Dup = new TreeNode(10);
//        rootDup.right = right1Dup;
//        TreeNode left2Dup = new TreeNode(1);
//        left1Dup.left = left2Dup;
//        TreeNode right2Dup = new TreeNode(6);
//        left1Dup.right = right2Dup;
//        TreeNode right3Dup = new TreeNode(14);
//        right1Dup.right = right3Dup;
//        TreeNode left4Dup = new TreeNode(4);
//        right2Dup.left = left4Dup;
//        TreeNode right4Dup = new TreeNode(7);
//        right2Dup.right = right4Dup;
//        TreeNode left5Dup = new TreeNode(13);
//        right3Dup.left = left5Dup;
//        /**
//         *                            8
//         *                          /  \
//         *                         3   10
//         *                        / \   \
//         *                       1  6   14
//         *                         / \  /
//         *                        4  7 13
//         */
//        TreeNode rootSys = new TreeNode(8);
//        TreeNode leftSys = new TreeNode(3);
//        rootSys.left = leftSys;
//        TreeNode rightSys = new TreeNode(3);
//        rootSys.right = rightSys;
//
//        Tree t = new Tree();
//        int minimumDepth = t.minDepth(root);
//        int maximumDepth = t.maxDepth(root);
////        ArrayList<Integer> preorder = t.preorderTraversalNoRecur(root);
////        ArrayList<Integer> inorder = t.inorderTraversalNoRecur(root);
////        ArrayList<Integer> postorder = t.postorderTraversalNoRecur(root);
////        ArrayList<Integer> levelorder = t.levelorderTraversalNoRecur(root);
//        boolean isIdentical = t.isIdentical(root, rootDup);
//        if (isIdentical) {
//            System.out.println("two trees are identical");
//        }
//        t.invertBinaryTree(rootDup);
//        boolean isTweakedIdentical = t.isTweakedIdentical(root, rootDup);
//        if (isTweakedIdentical) {
//            System.out.println("two trees are tweaked identical");
//        }
//        boolean isSys = t.isSymmetric(rootSys);
//        if (isSys) {
//            System.out.println("the tree is symmetric");
//        } else {
//            System.out.println("the tree is not symmetric");
//        }
//        boolean isCom1 = t.isComplete(root);
//        boolean isCom2 = t.isComplete(rootSys);
//        if (isCom1) {
//            System.out.println("the first tree is complete");
//        } else {
//            System.out.println("the first tree is not complete");
//        }
//        if (isCom2) {
//            System.out.println("the second tree is complete");
//        } else {
//            System.out.println("the second tree is not complete");
//        }
//        boolean isBal = t.isBalenced(root);
//        if (isBal) {
//            System.out.println("the tree is balanced");
//        } else {
//            System.out.println("the tree is not balanced");
//        }
//        int maxPathSum = t.maxPathSum(root);
//        System.out.println(maxPathSum);
////        String seri = t.serialize(root);
////        System.out.println(seri);
////        TreeNode newroot = t.deserialize(seri);
////        while (newroot != null) {
////            System.out.println(newroot.val);
////            newroot = newroot.right;
////        }
////        TreeNode removeroot = t.removeNode(root, new TreeNode(10));
////        ArrayList<Integer> inorder = t.inorderTraversalNoRecur(removeroot);
////        System.out.println("inorder traversal:");
////        for (int i = 0; i < inorder.size(); i++) {
////            System.out.println(inorder.get(i));
////        }
////        TreeNode addroot = t.insertNode(removeroot, new TreeNode(10));
////        ArrayList<Integer> inorder2 = t.inorderTraversalNoRecur(addroot);
////        System.out.println("inorder traversal:");
////        for (int i = 0; i < inorder2.size(); i++) {
////            System.out.println(inorder2.get(i));
////        }
//        ArrayList<Integer> searchrange = t.searchRange(root, 5, 10);
//        for (int i = 0; i < searchrange.size(); i++) {
//            System.out.println(searchrange.get(i));
//        }
////        System.out.println("minimum and maximum Depth:");
////        System.out.println(minimumDepth);
////        System.out.println(maximumDepth);
////        System.out.println("preorder traversal:");
////        for (int i = 0; i < preorder.size(); i++) {
////            System.out.println(preorder.get(i));
////        }
////        System.out.println("inorder traversal:");
////        for (int i = 0; i < inorder.size(); i++) {
////            System.out.println(inorder.get(i));
////        }
////        System.out.println("postorder traversal:");
////        for (int i = 0; i < postorder.size(); i++) {
////            System.out.println(postorder.get(i));
////        }
////        System.out.println("levelorder traversal:");
////        for (int i = 0; i < levelorder.size(); i++) {
////            System.out.println(levelorder.get(i));
////        }
////        t.invertBinaryTree(rootDup);
////        ArrayList<Integer> preorderInvert = t.preorderTraversalNoRecur(rootDup);
////        System.out.println("Inverted Tree preorder traversal:");
////        for (int i = 0; i < preorderInvert.size(); i++) {
////            System.out.println(preorderInvert.get(i));
////        }
//    }
}

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

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
        root = root.right;
        if (root.left != null) {
            return root.left;
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
     * Tree Reconstruction Problem: From sorted array, from pre and in order traversal, from post and in order traversal
     * Tree Serialization and deserialization
     * @param args
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return sortedArrayToBST(nums, 0, nums.length - 1) {
            
        }
    }





    public static void main(String args[]) {
        TreeNode root = new TreeNode(8);
        TreeNode left1 = new TreeNode(3);
        root.left = left1;
        TreeNode right1 = new TreeNode(10);
        root.right = right1;
        TreeNode left2 = new TreeNode(1);
        left1.left = left2;
        TreeNode right2 = new TreeNode(6);
        left1.right = right2;
        TreeNode right3 = new TreeNode(14);
        right1.right = right3;
        TreeNode left4 = new TreeNode(4);
        right2.left = left4;
        TreeNode right4 = new TreeNode(7);
        right2.right = right4;
        TreeNode left5 = new TreeNode(13);
        right3.left = left5;
        /**
         *                            8
         *                          /  \
         *                         3   10
         *                        / \   \
         *                       1  6   14
         *                         / \  /
         *                        4  7 13
          */
        TreeNode rootDup = new TreeNode(8);
        TreeNode left1Dup = new TreeNode(3);
        rootDup.left = left1Dup;
        TreeNode right1Dup = new TreeNode(10);
        rootDup.right = right1Dup;
        TreeNode left2Dup = new TreeNode(1);
        left1Dup.left = left2Dup;
        TreeNode right2Dup = new TreeNode(6);
        left1Dup.right = right2Dup;
        TreeNode right3Dup = new TreeNode(14);
        right1Dup.right = right3Dup;
        TreeNode left4Dup = new TreeNode(4);
        right2Dup.left = left4Dup;
        TreeNode right4Dup = new TreeNode(7);
        right2Dup.right = right4Dup;
        TreeNode left5Dup = new TreeNode(13);
        right3Dup.left = left5Dup;
        /**
         *                            8
         *                          /  \
         *                         3   10
         *                        / \   \
         *                       1  6   14
         *                         / \  /
         *                        4  7 13
         */
        TreeNode rootSys = new TreeNode(8);
        TreeNode leftSys = new TreeNode(3);
        rootSys.left = leftSys;
        TreeNode rightSys = new TreeNode(3);
        rootSys.right = rightSys;

        Tree t = new Tree();
        int minimumDepth = t.minDepth(root);
        int maximumDepth = t.maxDepth(root);
        ArrayList<Integer> preorder = t.preorderTraversalNoRecur(root);
        ArrayList<Integer> inorder = t.inorderTraversalNoRecur(root);
        ArrayList<Integer> postorder = t.postorderTraversalNoRecur(root);
        ArrayList<Integer> levelorder = t.levelorderTraversalNoRecur(root);
        boolean isIdentical = t.isIdentical(root, rootDup);
        if (isIdentical) {
            System.out.println("two trees are identical");
        }
        t.invertBinaryTree(rootDup);
        boolean isTweakedIdentical = t.isTweakedIdentical(root, rootDup);
        if (isTweakedIdentical) {
            System.out.println("two trees are tweaked identical");
        }
        boolean isSys = t.isSymmetric(rootSys);
        if (isSys) {
            System.out.println("the tree is symmetric");
        } else {
            System.out.println("the tree is not symmetric");
        }
        boolean isCom1 = t.isComplete(root);
        boolean isCom2 = t.isComplete(rootSys);
        if (isCom1) {
            System.out.println("the first tree is complete");
        } else {
            System.out.println("the first tree is not complete");
        }
        if (isCom2) {
            System.out.println("the second tree is complete");
        } else {
            System.out.println("the second tree is not complete");
        }
        boolean isBal = t.isBalenced(root);
        if (isBal) {
            System.out.println("the tree is balanced");
        } else {
            System.out.println("the tree is not balanced");
        }
        int maxPathSum = t.maxPathSum(root);
        System.out.println(maxPathSum);
//        System.out.println("minimum and maximum Depth:");
//        System.out.println(minimumDepth);
//        System.out.println(maximumDepth);
//        System.out.println("preorder traversal:");
//        for (int i = 0; i < preorder.size(); i++) {
//            System.out.println(preorder.get(i));
//        }
//        System.out.println("inorder traversal:");
//        for (int i = 0; i < inorder.size(); i++) {
//            System.out.println(inorder.get(i));
//        }
//        System.out.println("postorder traversal:");
//        for (int i = 0; i < postorder.size(); i++) {
//            System.out.println(postorder.get(i));
//        }
//        System.out.println("levelorder traversal:");
//        for (int i = 0; i < levelorder.size(); i++) {
//            System.out.println(levelorder.get(i));
//        }
//        t.invertBinaryTree(rootDup);
//        ArrayList<Integer> preorderInvert = t.preorderTraversalNoRecur(rootDup);
//        System.out.println("Inverted Tree preorder traversal:");
//        for (int i = 0; i < preorderInvert.size(); i++) {
//            System.out.println(preorderInvert.get(i));
//        }
    }
}

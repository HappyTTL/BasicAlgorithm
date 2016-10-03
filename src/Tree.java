import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Stack;

/**
 * Created by Tingting on 10/2/16.
 */
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
     * Tree traversal, preorder, inorder and postorder, both recursive and non-recursive
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
        Tree t = new Tree();
        int minimumDepth = t.minDepth(root);
        int maximumDepth = t.maxDepth(root);
        ArrayList<Integer> preorder = t.preorderTraversalNoRecur(root);
        ArrayList<Integer> inorder = t.inorderTraversalNoRecur(root);
        ArrayList<Integer> postorder = t.postorderTraversalNoRecur(root);
        System.out.println("minimum and maximum Depth:");
        System.out.println(minimumDepth);
        System.out.println(maximumDepth);
        System.out.println("preorder traversal:");
        for (int i = 0; i < preorder.size(); i++) {
            System.out.println(preorder.get(i));
        }
        System.out.println("inorder traversal:");
        for (int i = 0; i < inorder.size(); i++) {
            System.out.println(inorder.get(i));
        }
        System.out.println("postorder traversal:");
        for (int i = 0; i < postorder.size(); i++) {
            System.out.println(postorder.get(i));
        }
        t.invertBinaryTree(rootDup);
        ArrayList<Integer> preorderInvert = t.preorderTraversalNoRecur(rootDup);
        System.out.println("Inverted Tree preorder traversal:");
        for (int i = 0; i < preorderInvert.size(); i++) {
            System.out.println(preorderInvert.get(i));
        }
    }
}

import cv2
import numpy as np
import imageio
import pickle


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_rect = [(t[i][0] - r[0], t[i][1] - r[1]) for i in range(3)]

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, (r[2], r[3]))
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, (r[2], r[3]))

    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask


points1, points2, points3, points4 = [], [], [], []
img1, img2, img3, img4, img_combined = None, None, None, None, None

def click_event(event, x, y, flags, params):
    global img_combined, points1, points2, points3, points4

    if event == cv2.EVENT_LBUTTONDOWN:
        scale_factor = 0.5
        width = int(img1.shape[1] * scale_factor)
        height = int(img1.shape[0] * scale_factor)

        col = x // width
        row = y // height
        index = row * 2 + col
        relative_x = int(x % width / scale_factor)
        relative_y = int(y % height / scale_factor)

        if index == 0:
            points1.append((relative_x, relative_y))
        elif index == 1:
            points2.append((relative_x, relative_y))
        elif index == 2:
            points3.append((relative_x, relative_y))
        elif index == 3:
            points4.append((relative_x, relative_y))

        draw_points_on_images()

def draw_points_on_images():
    global img_combined, img1, img2, img3, img4

    img1_copy = cv2.resize(img1.copy(), (img1.shape[1] // 2, img1.shape[0] // 2))
    img2_copy = cv2.resize(img2.copy(), (img2.shape[1] // 2, img2.shape[0] // 2))
    img3_copy = cv2.resize(img3.copy(), (img3.shape[1] // 2, img3.shape[0] // 2))
    img4_copy = cv2.resize(img4.copy(), (img4.shape[1] // 2, img4.shape[0] // 2))

    for p in points1:
        cv2.circle(img1_copy, (int(p[0] // 2), int(p[1] // 2)), 5, (255, 0, 0), -1)
    for p in points2:
        cv2.circle(img2_copy, (int(p[0] // 2), int(p[1] // 2)), 5, (0, 255, 0), -1)
    for p in points3:
        cv2.circle(img3_copy, (int(p[0] // 2), int(p[1] // 2)), 5, (0, 0, 255), -1)
    for p in points4:
        cv2.circle(img4_copy, (int(p[0] // 2), int(p[1] // 2)), 5, (255, 255, 0), -1)

    top_row = np.hstack((img1_copy, img2_copy))
    bottom_row = np.hstack((img3_copy, img4_copy))
    img_combined = np.vstack((top_row, bottom_row))
    cv2.imshow('Image Morphing', img_combined)


def save_points(points1, points2, points3, points4, filename='saved_points.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((points1, points2, points3, points4), f)

def load_points(filename='saved_points.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return [], [], [], []

def display_points():
    global points1, points2, points3, points4, img_combined
    points1, points2, points3, points4 = load_points()
    scale_factor = 1

    img1_copy = img1.copy()
    img2_copy = img2.copy()
    img3_copy = img3.copy()
    img4_copy = img4.copy()

    scaled_points1 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points1]
    scaled_points2 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points2]
    scaled_points3 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points3]
    scaled_points4 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points4]

    for p in scaled_points1:
        cv2.circle(img1_copy, p, 5, (255, 0, 0), -1)
    for p in scaled_points2:
        cv2.circle(img2_copy, p, 5, (0, 255, 0), -1)
    for p in scaled_points3:
        cv2.circle(img3_copy, p, 5, (0, 0, 255), -1)
    for p in scaled_points4:
        cv2.circle(img4_copy, p, 5, (255, 255, 0), -1)

    refresh_images_with_copies(img1_copy, img2_copy, img3_copy, img4_copy)

def refresh_images_with_copies(img1_copy, img2_copy, img3_copy, img4_copy):
    new_size = (img1.shape[1] // 2, img1.shape[0] // 2)

    resized_img1 = cv2.resize(img1_copy, new_size)
    resized_img2 = cv2.resize(img2_copy, new_size)
    resized_img3 = cv2.resize(img3_copy, new_size)
    resized_img4 = cv2.resize(img4_copy, new_size)

    top_row = np.hstack((resized_img1, resized_img2))
    bottom_row = np.hstack((resized_img3, resized_img4))
    img_combined = np.vstack((top_row, bottom_row))

    cv2.imshow('Image Morphing', img_combined)

def refresh_images():
    global img_combined
    new_size = (img1.shape[1] // 2, img1.shape[0] // 2)

    resized_img1 = cv2.resize(img1, new_size)
    resized_img2 = cv2.resize(img2, new_size)
    resized_img3 = cv2.resize(img3, new_size)
    resized_img4 = cv2.resize(img4, new_size)

    top_row = np.hstack((resized_img1, resized_img2))
    bottom_row = np.hstack((resized_img3, resized_img4))
    img_combined = np.vstack((top_row, bottom_row))

    cv2.imshow('Image Morphing', img_combined)

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_indices = []
    triangles = subdiv.getTriangleList()
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        ind = []
        for pt in [pt1, pt2, pt3]:
            for idx, point in enumerate(points):
                if abs(point[0] - pt[0]) < 1 and abs(point[1] - pt[1]) < 1:
                    ind.append(idx)
                    break
        if len(ind) == 3:
            triangle_indices.append(ind)
    return triangle_indices

def perform_morphing(img1, img2, points1, points2, num_steps):

    frames = []
    alpha_steps = np.linspace(0, 1, num=num_steps)

    rect = (0, 0, img1.shape[1], img1.shape[0])
    tri_indices = calculate_delaunay_triangles(rect, points1)

    for alpha in alpha_steps:
        morphed_img = np.zeros_like(img1)
        for indices in tri_indices:
            x, y, z = indices
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [(1 - alpha) * np.array(points1[x]) + alpha * np.array(points2[x]),
                 (1 - alpha) * np.array(points1[y]) + alpha * np.array(points2[y]),
                 (1 - alpha) * np.array(points1[z]) + alpha * np.array(points2[z])]
            morph_triangle(img1, img2, morphed_img, t1, t2, t, alpha)
        frames.append(morphed_img)
    return frames

def perform_sequential_morphing():
    original_size = (img1.shape[1], img1.shape[0])
    frames = []
    frames += perform_morphing(img1, img2, points1, points2, 100)
    frames += perform_morphing(img2, img3, points2, points3, 100)
    frames += perform_morphing(img3, img4, points3, points4, 100)

    resized_frames = [cv2.resize(frame, original_size, interpolation=cv2.INTER_LINEAR) for frame in frames]

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized_frames]
    imageio.mimsave('morphing_sequence.gif', [np.uint8(frame) for frame in rgb_frames], 'GIF', duration=0.1)

    display_gif_bgr('morphing_sequence.gif')

def display_gif_bgr(gif_path):
    gif = imageio.get_reader(gif_path, format='gif', mode='I')
    for frame in gif:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("GIF Frame", bgr_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    global img1, img2, img3, img4, img_combined
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    img3 = cv2.imread('image3.jpg')
    img4 = cv2.imread('image4.jpg')

    if any(img is None for img in [img1, img2, img3, img4]):
        print("Error loading images.")
        return

    refresh_images()
    cv2.namedWindow('Image Morphing')
    cv2.setMouseCallback('Image Morphing', click_event)

    print("Select points on each image in sequence. Press 'm' to start morphing when done.")
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            if all(len(pts) > 0 for pts in [points1, points2, points3, points4]):
                perform_sequential_morphing()
            save_points(points1, points2, points3, points4)
            break
        elif k == ord('r'):
            display_points()
        elif k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
/*
 * ============================================================================
 * ROI MASK EXPORTER for QuPath
 * ============================================================================
 *
 * WHAT IT DOES:
 *   Takes one or more closed annotation regions in the current image and
 *   exports a binary mask TIFF at full resolution. Inside the annotation(s)
 *   is white (255), outside is black (0). Holes inside annotations are
 *   filled in (treated as solid regions).
 *
 *   Annotations whose NAME is "island" (case-insensitive) are treated as
 *   BACKGROUND instead of foreground: they are subtracted from the merged
 *   foreground mask, producing black holes wherever they fall. This lets
 *   you carve exclusions out of larger regions without re-drawing.
 *
 * HOW TO USE:
 *   1. Open your image in QuPath so it is the active/current image.
 *   2. Draw your annotation region(s) using the brush, wand, polygon, or
 *      any other annotation tool. Make sure each region is closed.
 *   3. To mark an annotation as background, set its NAME to "island0",
 *      "island1", "island2", ... (right-click annotation in the
 *      Annotations tab -> Set properties, or double-click the annotation).
 *      Case does not matter; "island_0" and "island-3" also work.
 *   4. (Optional) Select specific annotations in the Annotations tab or
 *      viewer if you only want to export a subset. If NOTHING is selected,
 *      the script will automatically use ALL annotations in the image.
 *   5. Open the script editor:  Automate -> Show script editor
 *   6. Paste this script in and press Run (Ctrl+R  /  Cmd+R).
 *   7. Watch the log pane at the bottom of the script editor for progress
 *      messages and the final output path.
 *
 * OUTPUT:
 *   A TIFF file named "<image_stem>_mask.tif" saved to your QuPath PROJECT
 *   folder (the directory shown as PROJECT_BASE_DIR). To change the filename
 *   or location, edit the `outputPath` line in the CONFIG block below.
 *
 * REQUIREMENTS:
 *   - The image must belong to a QuPath project (PROJECT_BASE_DIR must
 *     resolve). If you get a null path error, add the image to a project
 *     first:  File -> Project -> Create project, then drag the image in.
 *   - At least one non-island annotation must exist in the image.
 *
 * BEHAVIOR NOTES:
 *   - Multiple annotations are merged into a single mask. Overlapping or
 *     touching annotations are unioned safely.
 *   - Any holes inside annotations (interior rings) are filled with white.
 *     If you need holes preserved, remove the getExteriorRing() logic.
 *   - Island annotations are unioned together, then subtracted from the
 *     foreground union. Islands that fall outside any foreground region
 *     are simply ignored (nothing to subtract from).
 *   - Very large images (>500 megapixels) use a pyramidal OME-TIFF writer;
 *     smaller images are rasterized directly to a standard TIFF.
 *
 * TROUBLESHOOTING:
 *   - "No annotations found": you have not drawn any annotations yet.
 *   - "No foreground annotations": every annotation matches the island
 *     pattern (island0, island1, ...).
 *   - Mask appears empty or misaligned: check that annotations are on the
 *     same image plane (z-slice / timepoint) you expect.
 *   - Output path error: confirm the image is inside a QuPath project.
 * ============================================================================
 */

import qupath.lib.roi.RoiTools
import qupath.lib.roi.GeometryTools
import qupath.lib.objects.PathObjects
import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.regions.RegionRequest
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.Polygon
import org.locationtech.jts.geom.MultiPolygon
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LinearRing
import org.locationtech.jts.operation.union.UnaryUnionOp
import java.awt.image.BufferedImage
import java.awt.Graphics2D
import java.awt.Color
import java.awt.BasicStroke
import javax.imageio.ImageIO

def imageData = getCurrentImageData()
def server = imageData.getServer()

// ---------- CONFIG ----------
// Change this to redirect the output file. Default: <project folder>/<image>_mask.tif
def imageName = server.getMetadata().getName()
def stem = imageName.lastIndexOf('.') > 0 ? imageName[0..<imageName.lastIndexOf('.')] : imageName
def outputPath = buildFilePath(PROJECT_BASE_DIR, "${stem}_mask.tif")

// Names matching this pattern mark an annotation as background
// (case-insensitive). Matches "island0", "island1", "island12", etc.
// Also accepts an optional separator, so "island_0" / "island-3" work too.
def ISLAND_PATTERN = java.util.regex.Pattern.compile("(?i)^island[_-]?\\d+\$")
// ----------------------------

// Step 1: gather annotations.
// If the user has selected annotations, use only those.
// Otherwise fall back to using every annotation in the image.
def annotations = getSelectedObjects().findAll { it.isAnnotation() }
if (annotations.isEmpty()) {
    annotations = getAnnotationObjects()
    print "No selection - using all ${annotations.size()} annotation(s) in image."
}
if (annotations.isEmpty()) {
    print "No annotations found. Draw or select your annotations first."
    return
}

// Split annotations into foreground vs. island (background) groups
// based on the annotation NAME. Case-insensitive, null-safe.
def foregroundAnns = []
def islandAnns = []
annotations.each { ann ->
    def name = ann.getName()
    if (name != null && ISLAND_PATTERN.matcher(name.trim()).matches()) {
        islandAnns << ann
    } else {
        foregroundAnns << ann
    }
}
print "Processing ${foregroundAnns.size()} foreground annotation(s) and ${islandAnns.size()} island (background) annotation(s)."

if (foregroundAnns.isEmpty()) {
    print "No foreground annotations. Every annotation matches the island pattern - nothing to export."
    return
}

def gf = new GeometryFactory()

// Helper: given a list of annotations, extract exterior-ring-only polygons.
// Holes inside individual annotations are filled so each region is treated
// as solid. Returns a flat list of Polygon objects ready for unioning.
def extractFilledPolygons = { anns ->
    def polys = []
    anns.each { ann ->
        def roi = ann.getROI()
        def geom = roi.getGeometry()
        geom = geom.union()  // cleans/validates self-intersections on a single geom
        for (int i = 0; i < geom.getNumGeometries(); i++) {
            def g = geom.getGeometryN(i)
            if (g instanceof Polygon) {
                LinearRing shell = g.getExteriorRing()
                polys << gf.createPolygon(shell, null)
            }
        }
    }
    return polys
}

def foregroundPolys = extractFilledPolygons(foregroundAnns)
def islandPolys = extractFilledPolygons(islandAnns)

if (foregroundPolys.isEmpty()) {
    print "No polygons produced from foreground annotations."
    return
}

// Safely union all foreground polygons into one geometry.
// UnaryUnionOp handles disjoint, touching, and overlapping polygons correctly.
Geometry mergedForeground = foregroundPolys.size() == 1
    ? foregroundPolys[0]
    : UnaryUnionOp.union(foregroundPolys)

// Union islands (if any) and subtract from the foreground.
// Islands outside the foreground simply contribute nothing to the difference.
Geometry finalGeom = mergedForeground
if (!islandPolys.isEmpty()) {
    Geometry mergedIslands = islandPolys.size() == 1
        ? islandPolys[0]
        : UnaryUnionOp.union(islandPolys)
    print "Merged island geometry type: ${mergedIslands.getGeometryType()}, area: ${mergedIslands.getArea()}"
    finalGeom = mergedForeground.difference(mergedIslands)
}

print "Final geometry type: ${finalGeom.getGeometryType()}, area: ${finalGeom.getArea()}"

if (finalGeom.isEmpty()) {
    print "Final geometry is empty after island subtraction - nothing to write."
    return
}

def plane = foregroundAnns[0].getROI().getImagePlane()
def finalROI = GeometryTools.geometryToROI(finalGeom, plane)

// Step 2: rasterize at full resolution.
int width = server.getWidth()
int height = server.getHeight()
print "Image dimensions: ${width} x ${height}"

// Pick the rasterization strategy based on total pixel count.
// Large images need the pyramid writer to avoid running out of memory.
long pixelCount = (long) width * (long) height
if (pixelCount > 2_000_000_000L) {
    // ---- Large image path: pyramidal OME-TIFF ----
    print "Large image detected - using pyramid writer"

    // Temporarily add the final ROI as an annotation so LabeledImageServer
    // can render it. It is removed again after writing.
    def tempAnn = PathObjects.createAnnotationObject(finalROI, getPathClass("ROI_MASK"))
    addObject(tempAnn)

    def labelServer = new qupath.lib.images.servers.LabeledImageServer.Builder(imageData)
        .backgroundLabel(0, ColorTools.packRGB(0, 0, 0))
        .addLabel("ROI_MASK", 255, ColorTools.packRGB(255, 255, 255))
        .downsample(1.0)
        .multichannelOutput(false)
        .build()

    new OMEPyramidWriter.Builder(labelServer)
        .downsamples(1.0)
        .tileSize(512)
        .compression(OMEPyramidWriter.CompressionType.LZW)
        .parallelize()
        .build()
        .writePyramid(outputPath)

    removeObject(tempAnn, true)
    print "Saved mask to: ${outputPath}"
} else {
    // ---- Moderate-size path: draw directly to a BufferedImage ----
    // Graphics2D.fill() on a Java Shape derived from a JTS geometry with
    // holes uses the even-odd/non-zero winding rule correctly, so subtracted
    // island regions render as black automatically.
    print "Rasterizing directly"
    def img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    Graphics2D g2d = img.createGraphics()
    g2d.setColor(Color.BLACK)
    g2d.fillRect(0, 0, width, height)
    g2d.setColor(Color.WHITE)
    def shape = finalROI.getShape()
    g2d.fill(shape)
    g2d.dispose()

    def outFile = new File(outputPath)
    ImageIO.write(img, "TIFF", outFile)
    print "Saved mask to: ${outputPath}"
}
print "Done."
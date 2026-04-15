def project = getProject()
def outFile = new File(buildFilePath(PROJECT_BASE_DIR, 'image_names.csv'))
outFile.withWriter { writer ->
    for (entry in project.getImageList()) {
        def name = entry.getImageName().tokenize('/')[-1]
        name = name.replaceAll(/\.(tiff?|czi).*/, '.$1')
        def matcher = (entry.getImageName() =~ /ScanRegion(\d+)/)
        def region = matcher ? matcher[0][1] : '0'
        writer.writeLine("${name},${region}")
    }
}
println "Saved to: ${outFile.absolutePath}"
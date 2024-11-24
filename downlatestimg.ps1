# STEP ONE: Extract "token.encoded" from auth token
$response = Invoke-RestMethod -Uri "https://my.farm.bot/api/tokens" `
                              -Method Post `
                              -Headers @{"Content-Type"="application/json"} `
                              -Body '{"user":{"email":"skanda03prasad@gmail.com","password":"shaarvarp"}}'
$TOKEN = $response.token.encoded

# Check if token retrieval was successful
if (-not $TOKEN) {
    Write-Host "Failed to retrieve token. Exiting."
    exit 1
}

# STEP TWO: Get the number of objects from locations.json
try {
    $locationsContent = Get-Content -Raw -Path "locations.json"
    if (-not $locationsContent) {
        Write-Host "locations.json is empty. Exiting."
        exit 1
    }
    $locations = $locationsContent | ConvertFrom-Json
    $NUM_OBJECTS = $locations.Count
    Write-Host "Number of objects to download: $NUM_OBJECTS"
} catch {
    Write-Host "Error reading locations.json: $_"
    exit 1
}

# Check if number of objects is valid
if (-not $NUM_OBJECTS -or $NUM_OBJECTS -eq 0) {
    Write-Host "Invalid number of objects in locations.json. Exiting."
    exit 1
}

# STEP THREE: Create download directory
$downloadPath = "downloaded_images"
if (-not (Test-Path -Path $downloadPath)) {
    try {
        New-Item -Path $downloadPath -ItemType Directory -Force | Out-Null
        Write-Host "Created directory: $downloadPath"
    } catch {
        Write-Host "Failed to create download directory: $_"
        exit 1
    }
}

# STEP FOUR: Fetch and download images
try {
    # Fetch all images and sort by created_at in descending order
    $images = Invoke-RestMethod -Uri "https://my.farm.bot/api/images" `
                               -Headers @{"Authorization"="Bearer $TOKEN"}
    
    if (-not $images -or $images.Count -eq 0) {
        Write-Host "No images found in the FarmBot API."
        exit 1
    }

    # Sort images by created_at in descending order and take only the required number
    $sortedImages = $images | 
                    Sort-Object -Property created_at -Descending | 
                    Select-Object -First $NUM_OBJECTS

    # Download each image
    foreach ($image in $sortedImages) {
        $timestamp = $image.created_at
        $url = $image.attachment_url
        
        # Extract coordinates from meta field - no need to convert from JSON
        $x = [math]::Round($image.meta.x)
        $y = [math]::Round($image.meta.y)
        $z = [math]::Round($image.meta.z)
        
        # Convert timestamp to a filename-friendly format and add coordinates
        $filename = (Get-Date -Date $timestamp -Format "yyyy-MM-dd_HH-mm-ss") + "_x${x}_y${y}_z${z}.png"
        $filepath = Join-Path -Path $downloadPath -ChildPath $filename

        if (!(Test-Path -Path $filepath)) {
            Write-Host "Downloading: $filename"
            Invoke-WebRequest -Uri $url -OutFile $filepath
            Write-Host "Successfully downloaded: $filename (Coordinates: X:$x, Y:$y, Z:$z)"
        } else {
            Write-Host "File $filename already exists, skipping download."
        }
    }

    Write-Host "Download complete. Downloaded $($sortedImages.Count) images."

} catch {
    Write-Host "Error during image download process: $_"
    exit 1
}
using UnityEngine;

public class CameraManager : MonoBehaviour
{
    [SerializeField] private Camera worldCamera;
    [SerializeField] private Camera playerCamera;

    private void Start()
    {
        worldCamera.enabled = true;
        playerCamera.enabled = false;
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.E))
            SwitchCamera();
    }

    private void SwitchCamera()
    {
        worldCamera.enabled = !worldCamera.enabled;
        playerCamera.enabled = !playerCamera.enabled;
    }
}
